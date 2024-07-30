
import torch
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import numpy as np
from .preprocessing import DataTransformer


def loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]

def eval_step(
    model, 
    transformer, 
    loader, 
    loss_factor=2,
):
    epoch_loss = 0
    counter = 0
    with torch.no_grad():
        for id_, data in enumerate(loader):
            real = data[0].to(model.device)
            rec, sigmas, mu, logvar = model(real)
            loss_1, loss_2 = loss_function(
                rec, real, sigmas, mu, logvar,
                transformer.output_info_list, loss_factor
            )
            loss = loss_1 + loss_2
            model.decoder.sigma.data.clamp_(0.01, 1.0)
            epoch_loss += loss.item() * len(data)
            counter += len(data)

    epoch_loss /= counter
    return epoch_loss

def train(
    model, 
    transformer, 
    loader, 
    loss_factor=2,
    l2scale=1e-5,
    epochs=300,
    Optimizer=Adam,
    mlu_trainer=None,
    batch_size=512,
):
    model.train_history = []
    model.train()

    optimizerAE = Optimizer(
        model.parameters(),
        weight_decay=l2scale)

    for i in range(epochs):
        epoch_loss = 0
        counter = 0
        for id_, data in enumerate(loader):
            optimizerAE.zero_grad()
            real = data[0].to(model.device)
            rec, sigmas, mu, logvar = model(real)
            loss_1, loss_2 = loss_function(
                rec, real, sigmas, mu, logvar,
                transformer.output_info_list, loss_factor
            )
            loss = loss_1 + loss_2
            loss.backward()
            optimizerAE.step()
            model.decoder.sigma.data.clamp_(0.01, 1.0)
            epoch_loss += loss.item() * len(data)
            counter += len(data)

        epoch_loss /= counter

        model.train_history.append(epoch_loss)
                    
        if mlu_trainer:
            if mlu_trainer.should_step(i):
                pre_loss = eval_step(
                    model, 
                    transformer, 
                    loader, 
                    loss_factor=loss_factor,
                )

                total_mlu_loss = 0
                for _ in range(mlu_trainer.n_steps):
                    n_samples = mlu_trainer.n_samples
                    #batch_size = mlu_trainer.sample_batch_size
                    samples = sample(model=model, transformer=transformer, samples=n_samples, batch_size=batch_size, raw=True)
                    model.train()
                    mlu_loss, mlu_grad = mlu_trainer.step(samples, batch_size=batch_size)

                    total_mlu_loss += mlu_loss
                total_mlu_loss /= mlu_trainer.n_steps

                post_loss = eval_step(
                    model, 
                    transformer, 
                    loader, 
                    loss_factor=loss_factor,
                )
                mlu_trainer.log(
                    synthesizer_step=i,
                    train_loss=epoch_loss,
                    pre_loss=pre_loss,
                    mlu_loss=total_mlu_loss,
                    mlu_grad=mlu_grad,
                    post_loss=post_loss,
                    synthesizer_type="tvae",
                )
            else:
                mlu_trainer.log(
                    synthesizer_step=i,
                    train_loss=epoch_loss,
                    synthesizer_type="tvae",
                )
    if mlu_trainer:
        mlu_trainer.export_logs()

    return loss_1.item(), loss_2.item()


def sample(
    model, 
    transformer, 
    samples, 
    batch_size=500,
    raw=False,
):
    if not raw:
        model.eval()
    else:
        model.train()

    steps = samples // batch_size + 1
    data = []
    for _ in range(steps):
        mean = torch.zeros(batch_size, model.embedding_dim)
        std = mean + 1
        noise = torch.normal(mean=mean, std=std).to(model.device)
        fake, sigmas = model.decoder(noise)
        fake = torch.tanh(fake)
        if not raw:
            fake = fake.detach().cpu().numpy()
        data.append(fake)

    if not raw:
        data = np.concatenate(data, axis=0)
    else:
        data = torch.cat(data, dim=0)
    data = data[:samples]

    if raw:
        return data

    sigmas = sigmas.detach().cpu().numpy()
    if not transformer:
        return data
    return postprocess(
        transformer=transformer,
        data=data,
        sigmas=sigmas
    )

def preprocess(train_data, discrete_columns, transformer=None):
    if not transformer:
        transformer = DataTransformer()
        transformer.fit(train_data, discrete_columns)
    train_data = transformer.transform(train_data)
    return transformer, train_data

def postprocess(transformer, data, sigmas=None):
    return transformer.inverse_transform(data, sigmas)
