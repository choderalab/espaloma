# Relative alchemical free energy calculations

This is an example of using [perses](http://github.com/choderalab/perses) using espaloma to parameterize small molecules via []`openmmforcefields`](https://github.com/openmm/openmmforcefields)

* `tyk2/` - JACS tyk2 system

## Installing perses and espaloma

To install perses and espaloma together:
```bash
conda env create -n espaloma-perses -f espaloma-perses.yaml
```
To reproduce environment used in paper (on linux-64)
```bash
conda env create -n espaloma-perses -f espaloma-perses.export.yaml
```
