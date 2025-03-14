#/usr/env/bin python

'''
AMiGA variables that define the layout of Biolog PM Plates.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"

#### ACKNOWELDGMENT: ADOPTED FROM JAMES COLLINS. See NewPlateReader2018.py

Carbon1 = ["Negative Control",
         "L-Arabanose",
         "N-Acetyl-D-Glucosamine",
         "D-Saccharic Acid",
         "Succinic Acid",
         "D-Galactose",
         "L-Aspartic Acid",
         "L-Proline",
         "D-Alanine",
         "D-Trehalose",
         "D-Mannose",
         "Dulcitol",
         "D-Serine",
         "D-Sorbitol",
         "Glycerol",
         "L-Fucose",
         "D-Glucuronic Acid",
         "D-Gluconic Acid",
         "D,L-alpha-Glycerol-Phosphate",
         "D-Xylose",
         "L-Lactic Acid",
         "Formic Acid",
         "D-Mannitol",
         "L-Glutamic Acid",
         "D-Glucose-6-Phospate",
         "D-Galactonic Acid-gamma-Lactone",
         "D,L-Malic Acid",
         "D-Ribose",
         "Tween 20",
         "L-Rhamnose",
         "D-Fructose",
         "Acetic Acid",
         "alpha-D-Glucose",
         "Maltose",
         "D-Melibiose",
         "Thymidine",
         "L-Asparagine",
         "D-Aspartic Acid",
         "D-Glucosaminic Acid",
         "1,2-Propanediol",
         "Tween 40",
         "alpha-Keto-Glutaric Acid",
         "alpha-Keto-Butyric Acid",
         "alpha-Methyl-D-Galactoside",
         "alpha-D-Lactose",
         "Lactulose",
         "Sucrose",
         "Uridine",
         "L-Glutamine",
         "m-Tartaric Acid",
         "D-Glucose-1-Phosphate",
         "D-Fructose-6-Phosphate",
         "Tween 80",
         "alpha-Hydroxy Glutaric Acid-gamma-Lactone",
         "alpha-Hydroxy Butyric Acid", 
         "beta-Methyl-D-Glucoside",
         "Adonitol",
         "Maltotriose",
         "2-Deoxy Adenosine",
         "Adenosine",
         "Glycyl-L-Aspartic Acid",
         "Citric Acid",
         "m-Inositol",
         "D-Threonine",
         "Fumaric Acid",
         "Bromo Succinic Acid",
         "Propionic Acid",
         "Mucic Acid",
         "Glycolic Acid",
         "Glyoxylic Acid",
         "D-Cellobiose",
         "Inosine",
         "Glycyl-L-Glutamic Acid",
         "Tricarballylic Acid",
         "L-Serine",
         "L-Threonine",
         "L-Alanine",
         "L-Alanyl-Glycine",
         "Acetoacetic Acid",
         "N-Acetyl-beta-D-Mannosamine",
         "Mono Methyl Succinate",
         "Methyl Pyruvate",
         "D-Malic Acid",
         "L-Malic Acid",
         "Glycyl-L-Proline",
         "p-Hydroxy-Phenyl Acetic Acid",
         "m-Hydroxy-Phenyl Acetic Acid",
         "Tyramine",
         "D-Psicose",
         "L-Lyxose",
         "Glucuronamide",
         "Pyruvic Acid",
         "L-Galactonic Acid-gamma-Lactone",
         "D-Galacturonic Acid",
         "Phenylethylamine",
         "2-Aminoethanol"]

Carbon2 = ["Negative Control",
         "Chondroitin Sulfate C",
         "alpha-Cyclodextrin",
         "beta-Cyclodextrin",
         "gamma-Cyclodextrin",
         "Dextrin",
         "Gelatin",
         "Glycogen",
         "Inulin",
         "Laminarin",
         "Mannan",
         "Pectin",
         "N-Acetyl-D-Galactosamine",
         "N-Acetyl-Neuraminic-Acid",
         "beta-D-Allose",
         "Amygdalin",
         "D-Arabinose",
         "D-Arabitol",
         "L-Arabitol",
         "Arbutin",
         "2-Deoxy-D Ribose",
         "i-Erythritol",
         "D-Fucose",
         "3-0-beta-D-Galactopyranosyl-D Arabinose",
         "Gentiobiose",
         "L-Glucose",
         "Lactitol",
         "D-Melezitose",
         "Maltitol",
         "a-Methyl-D Glucoside",
         "beta-Methyl-D-Galactoside",
         "3-Methyl Glucose",
         "beta-Methyl-D-Glucuronic Acid",
         "alpha-Methyl-D-Mannoside",
         "beta-Methyl-D-Xyloside",
         "Palatinose",
         "D-Raffinose",
         "Salicin",
         "Sedoheptulosan",
         "L-Sorbose",
         "Stachyose",
         "D-Tagatose",
         "Turanose",
         "Xylitol",
         "N-Acetyl-D-Glucosaminitol",
         "gamma-Amino Butyric Acid",
         "delta-Amino Valeric Acid",
         "Butyric Acid",
         "Capric Acid",
         "Caproic Acid",
         "Citraconic Acid",
         "Citramalic Acid",
         "D-Glucosamine",
         "2-Hydroxy Benzoic Acid",
         "4-Hydroxy Benzoic Acid",
         "beta-Hydroxy Butyric Acid",
         "gamma-Hydroxy Butyric Acid",
         "a-Keto-Valeric Acid",
         "Itaconic Acid",
         "5-Keto-D-Gluconic Acid",
         "D-Lactic Acid Methyl Ester",
         "Malonic Acid",
         "Melibionic Acid",
         "Oxalic Acid",
         "Oxalomalic Acid",
         "Quinic Acid",
         "D-Ribono-1,4-Lactone",
         "Sebacic Acid",
         "Sorbic Acid",
         "Succinamic Acid",
         "D-Tartaric Acid",
         "L-Tartaric Acid",
         "Acetamide",
         "L-Alaninamide",
         "N-Acetyl-L-Glutamic Acid",
         "L-Arginine",
         "Glycine",
         "L-Histidine",
         "L-Homoserine",
         "Hydroxy-L-Proline",
         "L-Isoleucine",
         "L-Leucine",
         "L-Lysine",
         "L-Methionine",
         "L-Ornithine",
         "L-Phenylalanine",
         "L-Pyroglutamic Acid",
         "L-Valine",
         "D,L-Carnitine",
         "Sec-Butylamine",
         "D.L-Octopamine",
         "Putrescine",
         "Dihydroxy Acetone",
         "2,3-Butanediol",
         "2,3-Butanedione",
         "3-Hydroxy 2-Butanone"]

Nitrogen = ['Negative Control',
        'Ammonia',
        'Nitrite',
        'Nitrate',
        'Urea',
        'Biuret',
        'L-Alanine',
        'L-Arginine',
        'L-Asparagine',
        'L-Aspartic Acid',
        'L-Cysteine',
        'L-Glutamic Acid',
        'L-Glutamine',
        'Glycine',
        'L-Histidine',
        'L-Isoleucine',
        'L-Leucine',
        'L-Lysine',
        'L-Methionine',
        'L-Phenylalanine',
        'L-Proline',
        'L-Serine',
        'L-Threonine',
        'L-Tryptophan',
        'L-Tyrosine',
        'L-Valine',
        'D-Alanine',
        'D-Asparagine',
        'D-Aspartic Acid',
        'D-Glutamic Acid',
        'D-Lysine',
        'D-Serine',
        'D-Valine',
        'L-Citrulline',
        'L-Homoserine',
        'L-Ornithine',
        'N-Acetyl-L-Glutamic Acid',
        'N-Phthaloyl-L-Glutamic Acid',
        'L-Pyroglutamic Acid',
        'Hydroxylamine',
        'Methylamine',
        'N-Amylamine',
        'N-Butylamine',
        'Ethylamine',
        'Ethanolamine',
        'Ethylenediamine',
        'Putrescine',
        'Agmatine',
        'Histamine',
        'beta-Phenylethylamine',
        'Tyramine',
        'Acetamide',
        'Formamide',
        'Glucuronamide',
        'D,L-Lactamide',
        'D-Glucosamine',
        'D-Galactosamine',
        'D-Mannosamine',
        'N-Acetyl-D-Glucosamine',
        'N-Acetyl-D-Galactosamine',
        'N-Acetyl-D-Mannosamine',
        'Adenine',
        'Adenosine',
        'Cytidine',
        'Cytosine',
        'Guanine',
        'Guanosine',
        'Thymine',
        'Thymidine',
        'Uracil',
        'Uridine',
        'Inosine',
        'Xanthine',
        'Xanthosine',
        'Uric Acid',
        'Alloxan',
        'Allantoin',
        'Parabanic Acid',
        'D,L-alpha-Amino-N-Butyric Acid',
        'gamma-Amino-N-Butyric Acid',
        'g-Amino-N-Caproic Acid',
        'D,L-alpha-Amino Caprylic Acid',
        'delta-Amino-N-Valeric Acid',
        'alpha-Amino-N-Valeric Acid',
        'Ala-Asp',
        'Ala-Gln',
        'Ala-Glu',
        'Ala-Gly',
        'Ala-His',
        'Ala-Leu',
        'Ala-Thr',
        'Gly-Asn',
        'Gly-Gln',
        'Gly-Glu',
        'Gly-Met',
        'Met-Al']

PhosphorusAndSulfur = ["Negative Control",
        "Phosphate",
        "Pyrophosphate",
        "Trimetaphosphate",
        "Tripolyphosphate",
        "Triethyl Phosphate",
        "Hypophosphite",
        "Adenosine- 2'-monophosphate",
        "Adenosine- 3'-monophosphate",
        "Adenosine- 5'-monophosphate",
        "Adenosine- 2',3'-cyclic monophosphate",
        "Adenosine- 3',5'-cyclic monophosphate",
        "Thiophosphate",
        "Dithiophosphate",
        "D,L-alpha-Glycerol-Phosphate",
        "beta-Glycerol-Phosphate",
        "Carbamyl-Phosphate",
        "D-2-Phospho-Glyceric Acid",
        "D-3-Phospho-Glyceric Acid",
        "Guanosine-2'-monophosphate",
        "Guanosine-3'-monophosphate",
        "Guanosine-5'-monophosphate",
        "Guanosine-2',3'-cyclic monophosphate",
        "Guanosine-3',5'-cyclic monophosphate",
        "Phosphoenol Pyruvate",
        "Phospho-Glycolic Acid",
        "D-Glucose-1-Phosphate",
        "D-Glucose-6-Phosphate",
        "2-Deoxy-D-Glucose-6-Phosphate",
        "D-Glucosamine-6-Phosphate",
        "6-Phospho-Gluconic Acid",
        "Cytidine-2'-monophosphate",
        "Cytidine-3'-monophosphate",
        "Cytidine-5'-monophosphate",
        "Cytidine-2',3'-cyclic monophosphate",
        "Cytidine-3',5'-cyclic monophosphate",
        "D-Mannose-1-Phosphate",
        "D-Mannose-6-Phosphate",
        "Cysteamine-S-Phosphate",
        "Phospho-L-Arginine",
        "O-Phospho-D-Serine",
        "O-Phospho-L-Serine",
        "O-Phospho-L-Threonine",
        "Uridine-2'-monophosphate",
        "Uridine-3'-monophosphate",
        "Uridine-5'-monophosphate",
        "Uridine-2',3'-cyclic monophosphate",
        "Uridine-3',5'-cyclic monophosphate",
        "O-Phospho-D-Tyrosine",
        "O-Phospho-L-Tyrosine",
        "Phosphocreatine",
        "Phosphoryl Choline",
        "O-Phosphoryl-Ethanolamine",
        "Phosphono Acetic Acid",
        "2-Aminoethyl Phosphonic Acid",
        "Methylene Diphosphonic Acid",
        "Thymidine-3'-monophosphate",
        "Thymidine-5'-monophosphate",
        "Inositol Hexaphosphate",
        "Thymidine-3',5'-cyclic-monophosphate",
        "Negative Control",
        "Sulfate",
        "Thiosulfate",
        "Tetrathionate",
        "Thiophosphate",
        "Dithiophosphate",
        "L-Cysteine",
        "D-Cysteine",
        "L-Cysteinyl-Glycine",
        "L-Cysteic Acid",
        "Cysteamine",
        "L-Cysteine Sulfinic Acid",
        "N-Acetyl-L-Cysteine",
        "S-Methyl-L-Cysteine",
        "Cystathionine",
        "Lanthionine",
        "Glutathione",
        "D,L-Ethionine",
        "L-Methionine",
        "D-Methionine",
        "Glycyl-L-Methionine",
        "N-Acetyl-D,L-Methionine",
        "L- Methionine Sulfoxide",
        "L-Methionine Sulfone",
        "L-Djenkolic Acid",
        "Thiourea",
        "1-Thio-beta-D-Glucose",
        "D,L-Lipoamide",
        "Taurocholic Acid",
        "Taurine",
        "Hypotaurine",
        "p-Amino Benzene Sulfonic Acid",
        "Butane Sulfonic Acid",
        "2-Hydroxyethane Sulfonic Acid",
        "Methane Sulfonic Acid",
        "Tetramethylene Sulfone"]

PeptideNitrogen1 = ["Negative Control",
        "Positive Control: L-Glutamine",
        "Ala-Ala",
        "Ala-Arg",
        "Ala-Asn",
        "Ala-Glu",
        "Ala-Gly",
        "Ala-His",
        "Ala-Leu",
        "Ala-Lys",
        "Ala-Phe",
        "Ala-Pro",
        "Ala-Ser",
        "Ala-Thr",
        "Ala-Trp",
        "Ala-Tyr",
        "Arg-Ala",
        "Arg-Arg",
        "Arg-Asp",
        "Arg-Gln",
        "Arg-Glu",
        "Arg-Ile",
        "Arg-Leu",
        "Arg-Lys",
        "Arg-Met",
        "Arg-Phe",
        "Arg-Ser",
        "Arg-Trp",
        "Arg-Tyr",
        "Arg-Val",
        "Asn-Glu",
        "Asn-Val",
        "Asp-Asp",
        "Asp-Glu",
        "Asp-Leu",
        "Asp-Lys",
        "Asp-Phe",
        "Asp-Trp",
        "Asp-Val",
        "Cys-Gly",
        "Gln-Gln",
        "Gln-Gly",
        "Glu-Asp",
        "Glu-Glu",
        "Glu-Gly",
        "Glu-Ser",
        "Glu-Trp",
        "Glu-Tyr",
        "Glu-Val",
        "Gly-Ala",
        "Gly-Arg",
        "Gly-Cys",
        "Gly-Gly",
        "Gly-His",
        "Gly-Leu",
        "Gly-Lys",
        "Gly-Met",
        "Gly-Phe",
        "Gly-Pro",
        "Gly-Ser",
        "Gly-Thr",
        "Gly-Trp",
        "Gly-Tyr",
        "Gly-Val",
        "His-Asp",
        "His-Gly",
        "His-Leu",
        "His-Lys",
        "His-Met",
        "His-Pro",
        "His-Ser",
        "His-Trp",
        "His-Tyr",
        "His-Val",
        "Ile-Ala",
        "Ile-Arg",
        "Ile-Gln",
        "Ile-Gly",
        "Ile-His",
        "Ile-Ile",
        "Ile-Met",
        "Ile-Phe",
        "Ile-Pro",
        "Ile-Ser",
        "lle-Trp",
        "Ile-Tyr",
        "Ile-Val",
        "Leu-Ala",
        "Leu-Arg",
        "Leu-Asp",
        "Leu-Glu",
        "Leu-Gly",
        "Leu-Ile",
        "Leu-Leu",
        "Leu-Met",
        "Leu-Ph"]

PeptideNitrogen2 = ["Negative Control",
        "Positive Control: L-Glutamine",
        "Leu-Ser",
        "Leu-Trp",
        "Leu-Val",
        "Lys-Ala",
        "Lys-Arg",
        "Lys-Glu",
        "Lys-Ile",
        "Lys-Leu",
        "Lys-Lys",
        "Lys-Phe",
        "Lys-Pro",
        "Lys-Ser",
        "Lys-Thr",
        "Lys-Trp",
        "Lys-Tyr",
        "Lys-Val",
        "Met-Arg",
        "Met-Asp",
        "Met-Gln",
        "Met-Glu",
        "Met-Gly",
        "Met-His",
        "Met-Ile",
        "Met-Leu",
        "Met-Lys",
        "Met-Met",
        "Met-Phe",
        "Met-Pro",
        "Met-Trp",
        "Met-Val",
        "Phe-Ala",
        "Phe-Gly",
        "Phe-Ile",
        "Phe-Phe",
        "Phe-Pro",
        "Phe-Ser",
        "Phe-Trp",
        "Pro-Ala",
        "Pro-Asp",
        "Pro-Gln",
        "Pro-Gly",
        "Pro-Hyp",
        "Pro-Leu",
        "Pro-Phe",
        "Pro-Pro",
        "Pro-Tyr",
        "Ser-Ala",
        "Ser-Gly",
        "Ser-His",
        "Ser-Leu",
        "Ser-Met",
        "Ser-Phe",
        "Ser-Pro",
        "Ser-Ser",
        "Ser-Tyr",
        "Ser-Val",
        "Thr-Ala",
        "Thr-Arg",
        "Thr-Glu",
        "Thr-Gly",
        "Thr-Leu",
        "Thr-Met",
        "Thr-Pro",
        "Trp-Ala",
        "Trp-Arg",
        "Trp-Asp",
        "Trp-Glu",
        "Trp-Gly",
        "Trp-Leu",
        "Trp-Lys",
        "Trp-Phe",
        "Trp-Ser",
        "Trp-Trp",
        "Trp-Tyr",
        "Tyr-Ala",
        "Tyr-Gln",
        "Tyr-Glu",
        "Tyr-Gly",
        "Tyr-His",
        "Tyr-Leu",
        "Tyr-Lys",
        "Tyr-Phe",
        "Tyr-Trp",
        "Tyr-Tyr",
        "Val-Arg",
        "Val-Asn",
        "Val-Asp",
        "Val-Gly",
        "Val-His",
        "Val-Ile",
        "Val-Leu",
        "Val-Tyr",
        "Val-Val",
        "gamma-Glu-Gl"]

PeptideNitrogen3 = ["Negative Control",
        "Positive Control: L-Glutamine",
        "Ala-Asp",
        "Ala-Gln",
        "Ala-lle",
        "Ala-Met",
        "Ala-Val",
        "Asp-Ala",
        "Asp-Gln",
        "Asp-Gly",
        "Glu-Ala",
        "Gly-Asn",
        "Gly-Asp",
        "Gly-lle",
        "His-Ala",
        "His-Glu",
        "His-His",
        "Ile-Asn",
        "Ile-Leu",
        "Leu-Asn",
        "Leu-His",
        "Leu-Pro",
        "Leu-Tyr",
        "Lys-Asp",
        "Lys-Gly",
        "Lys-Met",
        "Met-Thr",
        "Met-Tyr",
        "Phe-Asp",
        "Phe-Glu",
        "Gln-Glu",
        "Phe-Met",
        "Phe-Tyr",
        "Phe-Val",
        "Pro-Arg",
        "Pro-Asn",
        "Pro-Glu",
        "Pro-lle",
        "Pro-Lys",
        "Pro-Ser",
        "Pro-Trp",
        "Pro-Val",
        "Ser-Asn",
        "Ser-Asp",
        "Ser-Gln",
        "Ser-Glu",
        "Thr-Asp",
        "Thr-Gln",
        "Thr-Phe",
        "Thr-Ser",
        "Trp-Val",
        "Tyr-lle",
        "Tyr-Val",
        "Val-Ala",
        "Val-Gln",
        "Val-Glu",
        "Val-Lys",
        "Val-Met",
        "Val-Phe",
        "Val-Pro",
        "Val-Ser",
        "beta-Ala-Ala",
        "beta-Ala-Gly",
        "beta-Ala-His",
        "Met-beta-Ala",
        "beta-Ala-Phe",
        "D-Ala-D-Ala",
        "D-Ala-Gly",
        "D-Ala-Leu",
        "D-Leu-D-Leu",
        "D-Leu-Gly",
        "D-Leu-Tyr",
        "gamma-Glu-Gly",
        "gamma-D-Glu-Gly",
        "Gly-D-Ala",
        "Gly-D-Asp",
        "Gly-D-Ser",
        "Gly-D-Thr",
        "Gly-D-Val",
        "Leu-beta-Ala",
        "Leu-D-Leu",
        "Phe-beta-Ala",
        "Ala-Ala-Ala",
        "D-Ala-Gly-Gly",
        "Gly-Gly-Ala",
        "Gly-Gly-D-Leu",
        "Gly-Gly-Gly",
        "Gly-Gly-lle",
        "Gly-Gly-Leu",
        "Gly-Gly-Phe",
        "Val-Tyr-Val",
        "Gly-Phe-Phe",
        "Leu-Gly-Gly",
        "Leu-Leu-Leu",
        "Phe-Gly-Gly",
        "Tyr-Gly-Gl"]


PM_LIST = [Carbon1,
          Carbon2,
          Nitrogen,
          PhosphorusAndSulfur,
          PeptideNitrogen1,
          PeptideNitrogen2,
          PeptideNitrogen3]

