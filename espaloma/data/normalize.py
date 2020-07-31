# =============================================================================
# IMPORTS
# =============================================================================
import abc

import torch

import espaloma as esp


# =============================================================================
# BASE CLASSES
# =============================================================================
class BaseNormalize(abc.ABC):
    """ Base class for normalizing operation.

    """

    def __init__(self):
        super(BaseNormalize, self).__init__()

    @abc.abstractmethod
    def _prepare(self):
        # NOTE:
        # `_norm` and `_unnorm` are assigned here
        raise NotImplementedError


# =============================================================================
# MODULE CLASSES
# =============================================================================
class DatasetNormalNormalize(BaseNormalize):
    """ Normalizing operation based on a dataset of molecules,
    assuming parameters having normal distribution.

    Parameters
    ----------
    dataset : `espaloma.data.dataset.Dataset`
        The dataset we base on to calculate the statistics of parameter
        distributions.

    Attributes
    ----------
    norm : normalize function

    unnorm : unnormalize function

    """

    def __init__(self, dataset):
        super(DatasetNormalNormalize, self).__init__()
        self.dataset = dataset
        self._prepare()

    def _prepare(self):
        """ Calculate the statistics from dataset """
        # grab the collection of graphs in the dataset, batched
        g = self.dataset.batch(self.dataset.graphs)

        self.statistics = {term: {} for term in ["n1", "n2", "n3", "n4"]}

        # calculate statistics
        for term in ["n1", "n2", "n3", "n4"]:  # loop through terms
            for key in g.nodes[term].data.keys():  # loop through parameters
                if not key.endswith("ref"):  # pass non-parameters
                    continue

                self.statistics[term][
                    key.replace("_ref", "_mean")
                ] = torch.mean(g.nodes[term].data[key], axis=0)

                self.statistics[term][key.replace("_ref", "_std")] = torch.std(
                    g.nodes[term].data[key], axis=0
                )

        # get normalize and unnormalize functions
        def norm(g):
            for term in ["n1", "n2", "n3", "n4"]:  # loop through terms
                for key in g.nodes[
                    term
                ].data.keys():  # loop through parameters
                    if not key.endswith("ref"):  # pass non-parameters
                        continue

                    g.nodes[term].data[key] = (
                        g.nodes[term].data[key]
                        - self.statistics[term][key.replace("_ref", "_mean")]
                    ) / self.statistics[term][key.replace("_ref", "_std")]

            return g

        def unnorm(g):
            for term in ["n1", "n2", "n3", "n4"]:  # loop through terms
                for key in g.nodes[
                    term
                ].data.keys():  # loop through parameters

                    if key + "_mean" in self.statistics[term]:

                        g.nodes[term].data[key] = (
                            g.nodes[term].data[key]
                            * self.statistics[term][key + "_std"]
                            + self.statistics[term][key + "_mean"]
                        )
                    #
                    # elif '_ref' in key \
                    #     and key.replace('_ref', '_mean')\
                    #     in self.statistics[term]:
                    #
                    #     g.nodes[term].data[key]\
                    #         = g.nodes[term].data[key]\
                    #             * self.statistics[term][
                    #                 key.replace('_ref', '_std')]\
                    #             + self.statistics[term][
                    #                 key.replace('_ref', '_mean')]

            return g

        # point normalize and unnormalize functions to `self`
        self.norm = norm
        self.unnorm = unnorm


class DatasetLogNormalNormalize(BaseNormalize):
    """ Normalizing operation based on a dataset of molecules,
    assuming parameters having log normal distribution.

    Parameters
    ----------
    dataset : `espaloma.data.dataset.Dataset`
        The dataset we base on to calculate the statistics of parameter
        distributions.

    Attributes
    ----------
    norm : normalize function

    unnorm : unnormalize function

    """

    def __init__(self, dataset):
        super(DatasetLogNormalNormalize, self).__init__()
        self.dataset = dataset
        self._prepare()

    def _prepare(self):
        """ Calculate the statistics from dataset """
        # grab the collection of graphs in the dataset, batched
        g = self.dataset.batch(self.dataset.graphs)

        self.statistics = {term: {} for term in ["n1", "n2", "n3", "n4"]}

        # calculate statistics
        for term in ["n1", "n2", "n3", "n4"]:  # loop through terms
            for key in g.nodes[term].data.keys():  # loop through parameters
                if not key.endswith("ref"):  # pass non-parameters
                    continue

                self.statistics[term][
                    key.replace("_ref", "_mean")
                ] = torch.mean(g.nodes[term].data[key].log(), axis=0)

                self.statistics[term][key.replace("_ref", "_std")] = torch.std(
                    g.nodes[term].data[key].log(), axis=0
                )

        # get normalize and unnormalize functions
        def norm(g):
            for term in ["n1", "n2", "n3", "n4"]:  # loop through terms
                for key in g.nodes[
                    term
                ].data.keys():  # loop through parameters
                    if not key.endswith("ref"):  # pass non-parameters
                        continue

                    g.nodes[term].data[key] = (
                        g.nodes[term].data[key].log()
                        - self.statistics[term][key.replace("_ref", "_mean")]
                    ) / self.statistics[term][key.replace("_ref", "_std")]

            return g

        def unnorm(g):
            for term in ["n1", "n2", "n3", "n4"]:  # loop through terms
                for key in g.nodes[
                    term
                ].data.keys():  # loop through parameters

                    if key + "_mean" in self.statistics[term]:

                        g.nodes[term].data[key] = torch.exp(
                            g.nodes[term].data[key]
                            * self.statistics[term][key + "_std"]
                            + self.statistics[term][key + "_mean"]
                        )
                    #
                    # elif '_ref' in key \
                    #     and key.replace('_ref', '_mean')\
                    #     in self.statistics[term]:
                    #
                    #     g.nodes[term].data[key]\
                    #         = torch.exp(
                    #             g.nodes[term].data[key]\
                    #                 * self.statistics[term][
                    #                     key.replace('_ref', '_std')]\
                    #                 + self.statistics[term][
                    #                     key.replace('_ref', '_mean')])

            return g

        # point normalize and unnormalize functions to `self`
        self.norm = norm
        self.unnorm = unnorm


# =============================================================================
# PRESETS
# =============================================================================
class ESOL100NormalNormalize(DatasetNormalNormalize):
    def __init__(self):
        super(ESOL100NormalNormalize, self).__init__(
            dataset=esp.data.esol(first=100).apply(
                esp.graphs.legacy_force_field.LegacyForceField(
                    "smirnoff99Frosst"
                ).parametrize,
                in_place=True,
            )
        )


class ESOL100LogNormalNormalize(DatasetLogNormalNormalize):
    def __init__(self):
        super(ESOL100LogNormalNormalize, self).__init__(
            dataset=esp.data.esol(first=100).apply(
                esp.graphs.legacy_force_field.LegacyForceField(
                    "smirnoff99Frosst"
                ).parametrize,
                in_place=True,
            )
        )


class NotNormalize(BaseNormalize):
    def __init__(self):
        super(NotNormalize).__init__()
        self._prepare()

    def _prepare(self):
        self.norm = lambda x: x
        self.unnorm = lambda x: x


class PositiveNotNormalize(BaseNormalize):
    def __init__(self):
        super(PositiveNotNormalize, self).__init__()
        self._prepare()

    def _prepare(self):

        # get normalize and unnormalize functions
        def norm(g):
            for term in ["n1", "n2", "n3", "n4"]:  # loop through terms
                for key in g.nodes[
                    term
                ].data.keys():  # loop through parameters
                    if not key.endswith("ref"):  # pass non-parameters
                        continue

                    g.nodes[term].data[key] = g.nodes[term].data[key].log()

            return g

        def unnorm(g):
            for term in ["n1", "n2", "n3", "n4"]:  # loop through terms
                for key in g.nodes[
                    term
                ].data.keys():  # loop through parameters
                    if not key + "_ref" in g.nodes[term].data:
                        continue

                    g.nodes[term].data[key] = torch.exp(
                        g.nodes[term].data[key]
                    )

            return g

        # point normalize and unnormalize functions to `self`
        self.norm = norm
        self.unnorm = unnorm
