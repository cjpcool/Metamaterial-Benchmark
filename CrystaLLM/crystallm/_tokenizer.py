import os
import re

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(THIS_DIR, "spacegroups.txt"), "rt") as f:
    SPACE_GROUPS = [sg.strip() for sg in f.readlines()]

DIGITS = [str(d) for d in list(range(10))]

KEYWORDS = [
    "_mechanical",
    "_youngs_modulus",
    "_shear_modulus",
    "_poisson_ratio",
    "_x",
    "_y",
    "_z",
    "_xy",
    "_xz",
    "_yz",
    "_yx",
    "_zx",
    "_zy",
    "_params",
    "_cell_length_a",
    "_cell_length_b",
    "_cell_length_c",
    "_cell_angle_alpha",
    "_cell_angle_beta",
    "_cell_angle_gamma",
    "_atom_site_fract_x",
    "_atom_site_fract_y",
    "_atom_site_fract_z", 
    "loop_",
    "data_",
    "lattice_",
    "_atom_index",
    "_edge_from_atom",
    "_edge_to_atom",
    "atom_"
]

EXTENDED_KEYWORDS = [
    "_atom_type_symbol",
    "_atom_type_electronegativity",  # does not exist in CIF spec
    "_atom_type_radius",  # does not exist in CIF spec
    "_atom_type_ionic_radius",  # does not exist in CIF spec
    "_atom_type_oxidation_number"
]

UNK_TOKEN = "<unk>"


class CIFTokenizer:
    def __init__(self):
        self._tokens = list()
        self._tokens.extend(self.digits())
        self._tokens.extend(self.keywords())
        self._tokens.extend(self.symbols())

        space_groups = list(self.space_groups())
        # Replace 'Pm' space group with 'Pm_sg' to disambiguate from atom 'Pm',
        #  or 'P1' with 'P1_sg' to disambiguate from atom 'P' and number '1'
        space_groups_sg = [sg+"_sg" for sg in space_groups]
        # self._tokens.extend(space_groups_sg)

        self._escaped_tokens = [re.escape(token) for token in self._tokens]
        self._escaped_tokens.sort(key=len, reverse=True)

        self._tokens_with_unk = list(self._tokens)
        self._tokens_with_unk.append(UNK_TOKEN)

        # a mapping from characters to integers
        self._token_to_id = {ch: i for i, ch in enumerate(self._tokens_with_unk)}
        self._id_to_token = {i: ch for i, ch in enumerate(self._tokens_with_unk)}
        # map the id of 'Pm_sg' back to 'Pm', or 'P1_sg' to 'P1',
        #  for decoding convenience
        # for sg in space_groups_sg:
        #     self._id_to_token[self.token_to_id[sg]] = sg.replace("_sg", "")

        print(f"The length of the tokenizer is {len(self._id_to_token)}")




    @staticmethod
    def digits():
        return DIGITS

    @staticmethod
    def keywords():
        kws = list(KEYWORDS)
        # kws.extend(EXTENDED_KEYWORDS)
        return kws

    @staticmethod
    def symbols():
        return ["x", "y", "z", ".", "(", ")", "+", "-", "/", "'", ",", " ", "\n"]

    @staticmethod
    def space_groups():
        return SPACE_GROUPS

    @property
    def token_to_id(self):
        return dict(self._token_to_id)

    @property
    def id_to_token(self):
        return dict(self._id_to_token)

    def encode(self, tokens):
        # encoder: take a list of tokens, output a list of integers
        return [self._token_to_id[t] for t in tokens]

    def decode(self, ids):
        # decoder: take a list of integers (i.e. encoded tokens), output a string
        return ''.join([self._id_to_token[i] for i in ids])

    def tokenize_cif(self, cif_string, single_spaces=True):
        # Preprocessing step to replace '_symmetry_space_group_name_H-M Pm'
        #  with '_symmetry_space_group_name_H-M Pm_sg',to disambiguate from atom 'Pm',
        #  or any space group symbol to avoid problematic cases, like 'P1'
        spacegroups = "|".join(SPACE_GROUPS)
        cif_string = re.sub(fr'(_symmetry_space_group_name_H-M *\b({spacegroups}))\n', r'\1_sg\n', cif_string)

        # Create a regex pattern by joining the escaped tokens with '|'
        token_pattern = '|'.join(self._escaped_tokens)

        # Add a regex pattern to match any sequence of characters separated by whitespace or punctuation
        full_pattern = f'({token_pattern}|\\w+|[\\.,;!?])'

        # Tokenize the input string using the regex pattern
        if single_spaces:
            cif_string = re.sub(r'[ \t]+', ' ', cif_string)
        tokens = re.findall(full_pattern, cif_string)

        # Replace unrecognized tokens with the unknown_token
        tokens = [token if token in self._tokens else UNK_TOKEN for token in tokens]

        return tokens
