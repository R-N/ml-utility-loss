from .autoencoder import LatentTAE
from .gan import LatentGAN
from sklearn.preprocessing import StandardScaler

import pickle
from tqdm import tqdm

import time
import torch
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

experiment_params = [
    {
        "best_ae" : 1000,
        "embedding_size": 64,
        "raw_csv_path": "./data/Adult.csv",
        "test_ratio": 0.20,
        "categorical_columns": ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'],
        "log_columns": [],
        "mixed_columns" : {'capital-loss': [0.0], 'capital-gain': [0.0]},
        "integer_columns" : ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'],
        "problem_type": {"Classification": 'income'}
    },
    {
        "best_ae": 1000,
        "embedding_size": 64,
        "raw_csv_path": "./data/Covtype.csv",
        "test_ratio": 0.20,
        "categorical_columns": ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
                                'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
                                'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
                                'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
                                'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
                                'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
                                'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
                                'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
                                'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
                                'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
                                'Soil_Type39', 'Soil_Type40', 'Cover_Type'],
        "log_columns": [],
        "mixed_columns": {"Hillshade_3pm": [0.0]},
        "integer_columns": ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                            'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                            'Horizontal_Distance_To_Fire_Points'],
        "problem_type": {"Classification": "Cover_Type"}
    },
    {
        "best_ae" : 1000,
        "embedding_size" : 96,
        "raw_csv_path": "./data/Credit.csv",
        "test_ratio": 0.20,
        "categorical_columns": ["Class"],
        "log_columns": ["Amount"],
        "mixed_columns": {},
        "integer_columns": ["Time"],
        "problem_type": {"Classification": "Class"}
    },
    {
        "best_ae" : 1000,
        "embedding_size": 32,
        "raw_csv_path" : "./data/Loan.csv",
        "test_ratio" : 0.20,
        "categorical_columns" : ["Family","Education","PersonalLoan","Securities Account","CD Account","Online","CreditCard"], # Note here ZIP Code is treated as numeric.  
        "log_columns" : [],
        "mixed_columns": {"Mortgage":[0.0]},
        "integer_columns": ["Age", "Experience", "Income","Mortgage"],
        "problem_type": {"Classification": "PersonalLoan"}
    }
]

def latent_gan_experiment(
        bottleneck=64,
        ae_epochs=1,
        ae_batch_size=512,
        gan_latent_dim=16,
        gan_epochs=1,
        gan_n_critic=2,
        gan_batch_size=512):

    for exp in experiment_params[:1]:
        exp = dict(exp)

        dataset_path = exp["raw_csv_path"]
        dataset_categories = exp["categorical_columns"]
        best_ae = exp["best_ae"]
        del exp["best_ae"]

        pickle_path = "./ae_pickles/" + exp['raw_csv_path'].replace("./data/", "").replace(".csv", f"_ae{exp['embedding_size']}_{best_ae}.pickle")

        print(f"Opening {pickle_path}")
        ae_pf = open(pickle_path, 'rb')
        ae = pickle.load(ae_pf)
        ae_pf.close()

        # EVALUATING AUTO-ENCODER
        latent_data = ae.get_latent_dataset() # could be loaded from file

        real_path = exp["raw_csv_path"]
        decoded_path = exp["raw_csv_path"].replace("./data/", "./data/decoded/").replace(".csv", f"_decoded{exp['embedding_size']}_test.csv")

        reconstructed_data = ae.decode(latent_data, batch=True)
        reconstructed_data.to_csv(decoded_path, index=False)

        real_path = real_path
        fake_paths = [ decoded_path ]

        sscaler = StandardScaler()
        sscaler.fit(latent_data)

        lat_normalized = sscaler.transform(latent_data)
        gan = LatentGAN(exp["embedding_size"], latent_dim=gan_latent_dim)

        def measure(x):
            df = gan.sample(len(latent_data), ae, sscaler)
            df.to_csv(dataset_path.replace(".csv", "_fake.csv"), index=False)
            print(df)

        gan.fit(lat_normalized, ae.train_data, ae.transformer, epochs=gan_epochs, batch_size=gan_batch_size, n_critic=gan_n_critic, callback=measure)

        gan_pf = open("./gan_pickles/" + exp['raw_csv_path'].replace("./data/", "").replace(".csv", f"_gan{gan_latent_dim}_{gan_epochs}.pickle"), 'wb')
        pickle.dump(gan, gan_pf)
        gan_pf.close()

def ae_experiment(
    epochs = 1,
    ae_batch_size=512,
):

    
    for exp in experiment_params[:1]:
        exp = dict(exp)

        best_ae = exp["best_ae"]
        del exp["best_ae"]
        
        print(f"Training on {exp['raw_csv_path']}")
        start_time = time.time()

        ae = LatentTAE(**exp)
        ae.fit(n_epochs=epochs, batch_size=ae_batch_size)
        time_to_train = time.time() - start_time
        print("--- %s seconds ---" % (time_to_train))

        ae_pf = open("./ae_pickles/" + exp['raw_csv_path'].replace("./data/", "").replace(".csv", f"_ae{exp['embedding_size']}_{epochs}.pickle"), 'wb')
        pickle.dump(ae, ae_pf)
        ae_pf.close()

        real_path = exp["raw_csv_path"]
        decoded_path = exp["raw_csv_path"].replace("./data/", "./data/decoded/").replace(".csv", f"_decoded{exp['embedding_size']}_{epochs}.csv")

        lat_data = ae.get_latent_dataset()

        reconstructed_data = ae.decode(lat_data, batch=True)

        reconstructed_data.to_csv(decoded_path, index=False)


if __name__ == "__main__":
    ae_experiment()
    latent_gan_experiment()