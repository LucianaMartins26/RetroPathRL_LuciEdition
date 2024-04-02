# from chemspipy import ChemSpider
# token = "0LOc74dO5KkadgoKQddvET1nsO2L8Z9D"

# cs = ChemSpider(token)

# building_blocks = pd.read_csv("/home/lmartins/BioNaviNP_LuciEdition/multistep/retro_star/dataset/bio_data/bio_building_blocks_all/building_blocks_new.csv")

# for smiles in building_blocks['mol']:
#     for result in cs.convert(smiles, 'SMILES', 'InChI'):
#         print(result, end = '')

from rdkit import Chem
import pandas as pd

def smiles_to_inchi(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Não foi possível converter o SMILES para uma molécula."
    inchi = Chem.MolToInchi(mol)
    return inchi

building_blocks = pd.read_csv("/home/lmartins/BioNaviNP_LuciEdition/multistep/retro_star/dataset/bio_data/bio_building_blocks_all/building_blocks_new.csv")

if __name__ == "__main__":
    for smiles in building_blocks['mol'][:-2]:
        print(smiles_to_inchi(smiles))