import unittest

import matplotlib
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.core import Spin

from sumo.plotting.bs_plotter import SBSPlotter

matplotlib.use("Agg")


class SanitiseLabelTestCase(unittest.TestCase):
    def test_sanitise_label(self):
        for label_in, label_out in (
            ("X", "X"),
            ("X 1", "X 1"),
            ("X @", "X "),
            ("X@", "X"),
            ("X@@@", "X"),
            ("HEX", "HEX"),
            ("HE@X", "HE@X"),
            ("HE@X 1", "HE@X 1"),
            ("@", None),
            ("@X", None),
            ("@HEX", None),
        ):
            self.assertEqual(SBSPlotter._sanitise_label(label_in), label_out)

    def test_sanitise_label_group(self):
        for label_in, label_out in (
            ("X", "X"),
            ("X 1", "X 1"),
            ("X @", "X "),
            ("X@", "X"),
            ("X@@@", "X"),
            ("HEX", "HEX"),
            ("HE@X", "HE@X"),
            ("HE@X 1", "HE@X 1"),
            ("@", None),
            ("@X", None),
            ("@HEX", None),
            (r"X$\mid$Y", r"X$\mid$Y"),
            (r"X$\mid$Y$\mid$Z", r"X$\mid$Y$\mid$Z"),
            (r"@X$\mid$Y$\mid$Z", r"Y$\mid$Z"),
            (r"X$\mid$@Y$\mid$Z", r"X$\mid$Z"),
            (r"X$\mid$@Z", r"X"),
            (r"X@$\mid$Y", r"X$\mid$Y"),
            (r"X@$\mid$Y@@", r"X$\mid$Y"),
            (r"@X@$\mid$Y@", r"Y"),
            (r"X@$\mid$@Y", r"X"),
            (r"@X@$\mid$@Y", None),
        ):
            self.assertEqual(SBSPlotter._sanitise_label_group(label_in), label_out)


class BandColourTestCase(unittest.TestCase):
    def test_semiconductor_edge_band_is_split_at_vbm(self):
        lattice = Lattice.cubic(1)
        structure = Structure(lattice, ["Si"], [[0, 0, 0]])
        kpoints = [[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]]
        labels = {r"\Gamma": (0, 0, 0), "X": (0.5, 0, 0)}
        bands = np.array([[0.0, 1.1, 0.8], [1.1, 1.2, 1.1]])

        bs = BandStructureSymmLine(
            kpoints,
            {Spin.up: bands},
            lattice.reciprocal_lattice_crystallographic,
            1.1,
            labels,
            coords_are_cartesian=False,
            structure=structure,
        )

        plt = SBSPlotter(bs).get_plot()
        colours = [line.get_color() for line in plt.gca().get_lines()]

        self.assertIn("C0", colours)
        self.assertIn("C1", colours)
