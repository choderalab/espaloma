import espaloma as esp


def run(name):
    g = esp.data.md17_utils.get_molecule(name)
    g.save(name)

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
    
