import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import os
from tqdm import tqdm

class PrepareData:

    def __init__(self):
        if 'ygong' in os.getcwd():
            self.filepath = "../data"
            self.dir_out = "../data/output"
        else:
            self.filepath = "/home/gong/Documents/Kaggle_July2019"
            self.dir_out = "/home/gong/Documents/Kaggle_July2019/output"

    def load_data(self):
        train = pd.read_csv(os.path.join(self.filepath, 'train.csv'))
        test = pd.read_csv(os.path.join(self.filepath, 'test.csv'))
        submit = pd.read_csv(os.path.join(self.filepath, 'sample_submission.csv'))
        structures = pd.read_csv(os.path.join(self.filepath, 'structures.csv'))
        scalar_coupling_contributions = pd.read_csv(os.path.join(self.filepath, 'scalar_coupling_contributions.csv'))

        print('Train dataset shape is -> rows: {} cols:{}'.format(train.shape[0], train.shape[1]))
        print('Test dataset shape is  -> rows: {} cols:{}'.format(test.shape[0], test.shape[1]))
        print('Sample submission dataset shape is  -> rows: {} cols:{}'.format(submit.shape[0], submit.shape[1]))
        print('Structures dataset shape is  -> rows: {} cols:{}'.format(structures.shape[0], structures.shape[1]))
        print('Scalar_coupling_contributions dataset shape is  -> rows: {} cols:{}'.format(
            scalar_coupling_contributions.shape[0],
            scalar_coupling_contributions.shape[1]))
        print('\n')

        ################################################################################################################
        # randomly take 10% of data for fast evaluation
        # Get only 10% of dataset for fast evaluation!
        # size = round(0.10 * train.shape[0])
        # train = train[:size]
        # test = test[:size]
        # submit = submit[:size]
        # structures = structures[:size]
        # scalar_coupling_contributions = scalar_coupling_contributions[:size]

        print('Train dataset shape is now rows: {} cols:{}'.format(train.shape[0], train.shape[1]))
        print('Test dataset shape is now rows: {} cols:{}'.format(test.shape[0], test.shape[1]))
        print('Sub dataset shape is now rows: {} cols:{}'.format(submit.shape[0], submit.shape[1]))
        print('Structures dataset shape is now rows: {} cols:{}'.format(structures.shape[0], structures.shape[1]))
        print('Scalar_coupling_contributions dataset shape is now rows: {} cols:{}'.format(
            scalar_coupling_contributions.shape[0],
            scalar_coupling_contributions.shape[1]))

        train = pd.merge(train, scalar_coupling_contributions, how='left',
                         left_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],
                         right_on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])

        return train, test, submit, structures

    def molecule_properties(self, structures):
        atomic_radius = {'H': 0.38, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71}
        fudge_factor = 0.05
        atomic_radius = {k: v + fudge_factor for k, v in atomic_radius.items()}
        print(atomic_radius)

        electronegativity = {'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98}

        atoms = structures['atom'].values
        atoms_en = [electronegativity[x] for x in tqdm(atoms)] # electronegrativity
        atoms_rad = [atomic_radius[x] for x in tqdm(atoms)]

        structures['EN'] = atoms_en
        structures['rad'] = atoms_rad

        return structures

    def bond_length(self, structures):
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

        print('Calculating the bonds')

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

        print('Counting and condensing bonds')

        bonds_numeric = [[i for i, x in enumerate(row) if x] for row in tqdm(bonds)]
        bond_lengths = [[dist for i, dist in enumerate(row) if i in bonds_numeric[j]] for j, row in
                        enumerate(tqdm(bond_dists))]
        bond_lengths_mean = [np.mean(x) for x in bond_lengths]
        n_bonds = [len(x) for x in bonds_numeric]

        # bond_data = {'bond_' + str(i):col for i, col in enumerate(np.transpose(bonds))}
        # bond_data.update({'bonds_numeric':bonds_numeric, 'n_bonds':n_bonds})

        bond_data = {'n_bonds': n_bonds, 'bond_lengths_mean': bond_lengths_mean}
        bond_df = pd.DataFrame(bond_data)
        structures = structures.join(bond_df)
        # save data
        structures.to_csv(os.path.join(self.filepath, 'molecular_structure.csv'))

        return structures

    def map_atom_info(self, df, atom_idx, structures):

        df = pd.merge(df, structures, how='left',
                      left_on=['molecule_name', f'atom_index_{atom_idx}'],
                      right_on=['molecule_name', 'atom_index'])

        df = df.rename(columns={'atom': f'atom_{atom_idx}',
                                'x': f'x_{atom_idx}',
                                'y': f'y_{atom_idx}',
                                'z': f'z_{atom_idx}'})
        return df

    def del_cols(self, df, cols):
        del_cols_list_ = [l for l in cols if l in df]
        df = df.drop(del_cols_list_, axis=1)
        return df

    def encode_categoric_single(self, df):
        lbl = LabelEncoder()
        cat_cols = []
        try:
            cat_cols = df.describe(include=['O']).columns.tolist()
            for cat in cat_cols:
                df[cat] = lbl.fit_transform(list(df[cat].values))
        except Exception as e:
            print('error: ', str(e))

        return df

    def prepare_train_test(self, train, test, structures):
        train = self.map_atom_info(train, 0, structures)
        train = self.map_atom_info(train, 1, structures)

        test = self.map_atom_info(test, 0, structures)
        test = self.map_atom_info(test, 1, structures)

        del_cols_list = ['id', 'molecule_name', 'sd', 'pso', 'dso']
        train = self.del_cols(train, del_cols_list)
        test = self.del_cols(test, del_cols_list)

        train = self.encode_categoric_single(train)
        test = self.encode_categoric_single(test)

        train.to_csv(os.path.join(self.dir_out, "_train.csv"), index=False)
        print("Saved training dataset to {}".format(os.path.join(self.dir_out, "_train.csv")))

        test.to_csv(os.path.join(self.dir_out, "_test.csv"))
        print("Saved test dataset to {}".format(os.path.join(self.dir_out, "_test.csv"), index=False))


if __name__ == "__main__":

    features = PrepareData()
    train, test, submit, structures = features.load_data()
    structures = features.molecule_properties(structures)
    structures = features.bond_length(structures)
    features.prepare_train_test(train, test, structures)
