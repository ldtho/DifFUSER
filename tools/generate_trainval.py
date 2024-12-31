import pickle

import numpy as np


def split_data(train_data, val_data, val_percentage=5):
    """
    Merges train and val data, then splits them into new train and val sets based on the given val_percentage.

    Args:
    - train_data (list): List of training data.
    - val_data (list): List of validation data.
    - val_percentage (int, optional): Percentage of data to be used for validation. Defaults to 5.

    Returns:
    - tuple: new_train_data, new_val_data
    """
    # Calculate the sampling interval based on the desired validation percentage
    interval = int(100 / val_percentage)

    # Merge the two sets
    combined_data = train_data + val_data

    # Convert to numpy array for efficient slicing
    combined_data_np = np.array(combined_data)

    # Sample based on the interval for validation
    new_val = combined_data_np[::interval].tolist()

    # The rest for training
    new_train = combined_data_np[np.arange(len(combined_data_np)) % interval != 0].tolist()

    return new_train, new_val


if __name__ == '__main__':
    # load nuscenes_infos_train.pkl
    with open('../data/nuscenes/nuscenes_infos_train.pkl', 'rb') as f:
        nuscenes_infos_train = pickle.load(f)['infos']
    # load nuscenes_infos_val.pkl
    with open('../data/nuscenes/nuscenes_infos_val.pkl', 'rb') as f:
        nuscenes_infos_val = pickle.load(f)['infos']

    # Merge the two sets
    combined_data = nuscenes_infos_train + nuscenes_infos_val
    combined_data = {
        'infos': combined_data,
        'metadata':{'version': 'v1.0-trainval'},
    }
    with open('../data/nuscenes/nuscenes_infos_trainval.pkl', 'wb') as f:
        pickle.dump(combined_data, f)

    ## Desired validation percentage
    # desired_val_percentage = 5  # You can change this value as needed
    #
    # new_train, new_val, = split_data(nuscenes_infos_train, nuscenes_infos_val, desired_val_percentage)
    #
    # new_train = {
    #     'infos': new_train,
    #     'metadata':{'version': 'v1.0-trainval'},
    # }
    # new_val = {
    #     'infos': new_val,
    #     'metadata':{'version': 'v1.0-trainval'},
    # }
    #
    # # Save the new train and val sets to pickle files
    # with open('../data/nuscenes/new_nuscenes_infos_train.pkl', 'wb') as f:
    #     pickle.dump(new_train, f)
    #
    # with open('../data/nuscenes/new_nuscenes_infos_val.pkl', 'wb') as f:
    #     pickle.dump(new_val, f)

