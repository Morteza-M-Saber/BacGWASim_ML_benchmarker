"""Recieves a GCTA simulated phenotype file and transform it into pyseer format"""


def get_options():
    import argparse

    description = (
        "Recieves a GCTA simulated phenotype file and transform it into pyseer format"
    )
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("phenotype", help="GCTA simulated phenotypes")
    parser.add_argument("--phenType", help="cc for case-control, quant for continuous")
    parser.add_argument("--output", help="Output file")

    return parser.parse_args()


options = get_options()


def PhenoFormater(phenotype, phenType, output):
    # Transform the gatc-simulated phenotypes to the format usable by pyseer
    txt = open(output, "w")
    txt.write("Samples\tPhenotype\n")
    with open(phenotype, "r") as file:
        line = file.readline()
        while line:
            name = line.split()[1] + "_"
            if phenType == "quant":
                txt.write(
                    "%s\t%s\n"
                    % (name, line.split()[2].replace("1", "0").replace("2", "1"))
                )
            elif phenType == "cc":
                txt.write(
                    "%s\t%s\n"
                    % (
                        name,
                        line.split()[2]
                        .replace("1", "0")
                        .replace("2", "1")
                        .replace("-9", "NA"),
                    )
                )
            line = file.readline()


PhenoFormater(options.phenotype, options.phenType, options.output)
