import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def load_data(filepath):
    train = pd.read_csv(os.path.join(filepath, 'train.csv'))
    test = pd.read_csv(os.path.join(filepath, 'test.csv'))
    submit = pd.read_csv(os.path.join(filepath, 'sample_submission.csv'))
    structures = pd.read_csv(os.path.join(filepath, 'structures.csv'))

    print('Train dataset shape is -> rows: {} cols:{}'.format(train.shape[0], train.shape[1]))
    print('Test dataset shape is  -> rows: {} cols:{}'.format(test.shape[0], test.shape[1]))
    print('Sample submission dataset shape is  -> rows: {} cols:{}'.format(submit.shape[0], submit.shape[1]))
    print('Structures dataset shape is  -> rows: {} cols:{}'.format(structures.shape[0], structures.shape[1]))
    print('\n')

    return train, test, submit, structures

def molecule_properties(structures):
    atomic_radius = {'H': 0.38, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71}
    fudge_factor = 0.05
    atomic_radius = {k: v + fudge_factor for k, v in atomic_radius.items()}
    print(atomic_radius)
    print('\n')

    electronegativity = {'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98}

    atoms = structures['atom'].values
    atoms_en = [electronegativity[x] for x in tqdm(atoms)] # electronegrativity
    atoms_rad = [atomic_radius[x] for x in tqdm(atoms)]

    structures['EN'] = atoms_en
    structures['rad'] = atoms_rad

    return structures

def bond_length(structures):
    i_atom = structures['atom_index'].values
    p = structures[['x', 'y', 'z']].values
    p_compare = p
    m = structures['molecule_name'].values
    m_compare = m
    r = structures['rad'].values
    r_compare = r

    source_row = np.arange(len(structures))
    max_atoms = 28

    bonds = np.zeros((len(structures) + 1, max_atoms + 1), dtype=np.int8)
    bond_dists = np.zeros((len(structures) + 1, max_atoms + 1), dtype=np.float32)

    print('\nCalculating the bonds')

    for i in tqdm(range(max_atoms - 1)):
        p_compare = np.roll(p_compare, -1, axis=0)
        m_compare = np.roll(m_compare, -1, axis=0)
        r_compare = np.roll(r_compare, -1, axis=0)

        # Are we still comparing atoms in the same molecule?
        mask = np.where(m == m_compare, 1, 0)
        dists = np.linalg.norm(p - p_compare, axis=1) * mask
        r_bond = r + r_compare

        bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

        source_row = source_row
        # Note: Will be out of bounds of bonds array for some values of i
        target_row = source_row + i + 1
        # If invalid target, write to dummy row
        target_row = np.where(np.logical_or(target_row > len(structures), mask == 0), len(structures),
                              target_row)  # If invalid target, write to dummy row

        source_atom = i_atom
        # Note: Will be out of bounds of bonds array for some values of i
        target_atom = i_atom + i + 1
        # If invalid target, write to dummy col
        target_atom = np.where(np.logical_or(target_atom > max_atoms, mask == 0), max_atoms,
                               target_atom)

        bonds[(source_row, target_atom)] = bond
        bonds[(target_row, source_atom)] = bond
        bond_dists[(source_row, target_atom)] = dists
        bond_dists[(target_row, source_atom)] = dists

    bonds = np.delete(bonds, axis=0, obj=-1)  # Delete dummy row
    bonds = np.delete(bonds, axis=1, obj=-1)  # Delete dummy col
    bond_dists = np.delete(bond_dists, axis=0, obj=-1)  # Delete dummy row
    bond_dists = np.delete(bond_dists, axis=1, obj=-1)  # Delete dummy col

    print('\nCounting and condensing bonds')

    bonds_numeric = [[i for i, x in enumerate(row) if x] for row in tqdm(bonds)]
    bond_lengths = [[dist for i, dist in enumerate(row) if i in bonds_numeric[j]] for j, row in
                    enumerate(tqdm(bond_dists))]
    bond_lengths_mean = [np.mean(x) for x in bond_lengths]
    n_bonds = [len(x) for x in bonds_numeric]

    # bond_data = {'n_bonds': n_bonds, 'bond_lengths_mean': bond_lengths_mean}
    # bond_df = pd.DataFrame(bond_data)
    # structures = structures.join(bond_df)
    structures['n_bonds'] = n_bonds
    structures['bond_lengths_mean'] = bond_lengths_mean

    return structures

def map_atom_info(df, atom_idx, structures):

    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])

    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df


def calculate_dist(df):
    p_0 = df[['x_0', 'y_0', 'z_0']].values
    p_1 = df[['x_1', 'y_1', 'z_1']].values
    df['dist'] = np.linalg.norm(p_0 - p_1, axis=1)
    return df


def get_intervene_bonds(df):
    J_type = df['type'].values
    intervene_bonds = [int(item[0]) for item in tqdm(J_type)]
    df['intervene_bonds'] = intervene_bonds
    return df


def get_features(filepath):

    # load files
    train, test, submit, structures = load_data(filepath)

    # add atomic radii and electronegativity
    structures = molecule_properties(structures)

    # calculate average bond length to each atom
    structures = bond_length(structures)

    # map structures to train and test
    print('Mapping the structure to train & test ...')
    train = map_atom_info(train, 0, structures)
    train = map_atom_info(train, 1, structures)
    test = map_atom_info(test, 0, structures)
    test = map_atom_info(test, 1, structures)

    print("Calculate the distance between two atoms")
    train = calculate_dist(train)
    test = calculate_dist(test)

    # extract the number of intervening bonds
    print('Extracting number of intervene bonds from type')
    train = get_intervene_bonds(train)
    test = get_intervene_bonds(test)

    # get polarity (difference of electronegativity)
    train['polarity'] = abs(train['EN_x'] - train['EN_y'])
    test['polarity'] = abs(test['EN_x'] - test['EN_y'])

    return train, test


if __name__ == "__main__":

    if 'ygong' in os.getcwd():
        filepath = "../data"
        dir_out = "../data/output"
    else:
        filepath = "/home/gong/Documents/Kaggle_July2019/data"
        dir_out = "/home/gong/Documents/Kaggle_July2019/output"

    train, test = get_features(filepath)
