import os, sys
import argparse
import pandas as pd
import json
import random

from EVE import VAE_model
from utils import data_utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--MSA_list', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', type=str, help='Location where weights for each sequence in the MSA will be stored')
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--VAE_checkpoint_location', type=str, help='Location where VAE model checkpoints will be stored')
    parser.add_argument('--model_name_suffix', default='Jan1', type=str, help='model checkpoint name will be the protein name followed by this suffix')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--output_base_dir', type=str, default='experiments', help='Base directory for output files')
    parser.add_argument('--training_logs_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--latent_dim', type=int, default=50, help='Dimensionality of the latent space')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--use_validation', action='store_true', help='Whether to use a validation set during training')
    args = parser.parse_args()

    # Debug validation flag
    print(f"Validation flag is set to: {args.use_validation}")

    mapping_file = pd.read_csv(args.MSA_list)
    protein_name = mapping_file['protein_name'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['msa_location'][args.protein_index]
    print("Protein name: "+str(protein_name))
    print("MSA file: "+str(msa_location))

    if args.theta_reweighting is not None:
        theta = args.theta_reweighting
    else:
        try:
            theta = float(mapping_file['theta'][args.protein_index])
        except:
            theta = 0.2
    print("Theta MSA re-weighting: "+str(theta))

    data = data_utils.MSA_processing(
            MSA_location=msa_location,
            theta=theta,
            use_weights=True,
            preprocess_MSA=False,
            weights_location=args.MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'
    )

    experiment_id = f"seed{args.seed}_theta{args.theta_reweighting}_ld{args.latent_dim}_lr{args.learning_rate}"
    output_dir = os.path.join(args.output_base_dir, experiment_id)

    model_params = json.load(open(args.model_parameters_location))

    # Update encoder/decoder latent dimension size
    model_params["encoder_parameters"]["z_dim"] = args.latent_dim
    model_params["decoder_parameters"]["z_dim"] = args.latent_dim
    model_params["training_parameters"]["learning_rate"] = args.learning_rate

    # Use the validation flag from command line arguments
    if args.use_validation:
        print("Enabling validation set (20% of data)")
        model_params["training_parameters"]["use_validation_set"] = True
        model_params["training_parameters"]["validation_set_pct"] = 0.2
    else:
        print("Validation set disabled")
        model_params["training_parameters"]["use_validation_set"] = False
        model_params["training_parameters"]["validation_set_pct"] = 0
    
    # Always disable test set for consistency
    model_params["training_parameters"]["use_test_set"] = False
    model_params["training_parameters"]["test_set_pct"] = 0

    model_params["training_parameters"]["num_training_steps"] = 200000

    model_name = f"{args.model_name_suffix}_{experiment_id}"
    print("Model name: " + model_name)

    model = VAE_model.VAE_model(
                    model_name=model_name,
                    data=data,
                    encoder_parameters=model_params["encoder_parameters"],
                    decoder_parameters=model_params["decoder_parameters"],
                    random_seed=args.seed
    )
    # Make sure to set CUDA_VISIBLE_DEVICES accordingly
    model = model.to(model.device)

    os.makedirs(output_dir, exist_ok=True)
    model_params["training_parameters"]['training_logs_location'] = os.path.join(output_dir, 'logs')
    model_params["training_parameters"]['model_checkpoint_location'] = os.path.join(output_dir, 'checkpoints')
    os.makedirs(model_params["training_parameters"]['training_logs_location'], exist_ok=True)
    os.makedirs(model_params["training_parameters"]['model_checkpoint_location'], exist_ok=True)

    # model_params["training_parameters"]['training_logs_location'] = args.training_logs_location
    # model_params["training_parameters"]['model_checkpoint_location'] = args.VAE_checkpoint_location

    print("Starting to train model: " + model_name)
    model.train_model(data=data, training_parameters=model_params["training_parameters"])

    print("Saving model: " + model_name)
    model.save(model_checkpoint=model_params["training_parameters"]['model_checkpoint_location']+os.sep+model_name+"_final", 
                encoder_parameters=model_params["encoder_parameters"], 
                decoder_parameters=model_params["decoder_parameters"], 
                training_parameters=model_params["training_parameters"]
    )

    print("Training completed for experiment: " + experiment_id)