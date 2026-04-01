import os
import shutil
import tempfile
import unittest

try:
    from importlib.resources import files as ilr_files
except ImportError:  # Python < 3.9
    from importlib_resources import files as ilr_files

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
from pymatgen.electronic_structure.core import Spin

from sumo.cli.bandplot import _with_abacus_band_partner, bandplot
from sumo.cli.dosplot import (
    _find_abacus_dos_files,
    _with_abacus_spin_partner,
    dosplot,
)
from sumo.cli.kgen import kgen
from sumo.io.abacus import (
    band_structure,
    read_dos,
    read_efermi,
    read_kpoint_labels,
    read_structure,
    write_kpoint_files,
)


class AbacusIoTestCase(unittest.TestCase):
    def setUp(self):
        base = os.path.join(ilr_files("tests"), "data", "abacus")

        self.latest_band_dir = os.path.join(base, "latest", "band")
        self.latest_band_spin_dir = os.path.join(base, "latest", "band_spin")
        self.latest_dos_dir = os.path.join(base, "latest", "dos")
        self.latest_dos_spin_dir = os.path.join(base, "latest", "dos_spin")
        self.lts_band_dir = os.path.join(base, "lts", "band")
        self.lts_band_spin_dir = os.path.join(base, "lts", "band_spin")
        self.lts_dos_dir = os.path.join(base, "lts", "dos")
        self.lts_dos_spin_dir = os.path.join(base, "lts", "dos_spin")

        self.band_text_file = os.path.join(self.latest_band_dir, "band.txt")
        self.band_kpt = os.path.join(self.latest_band_dir, "KPT_BANDS")
        self.band_stru = os.path.join(self.latest_band_dir, "STRU")
        self.band_log = os.path.join(self.latest_band_dir, "running_nscf.log")

        self.lts_band_file = os.path.join(self.lts_band_dir, "BANDS_1.dat")
        self.lts_band_text_file = os.path.join(self.lts_band_dir, "band.txt")
        self.lts_band_kpt = os.path.join(self.lts_band_dir, "KPT_BANDS")
        self.lts_band_stru = os.path.join(self.lts_band_dir, "STRU")
        self.lts_band_log = os.path.join(self.lts_band_dir, "running_nscf.log")

        self.latest_spin_band_files = [
            os.path.join(self.latest_band_spin_dir, "bands1.txt"),
            os.path.join(self.latest_band_spin_dir, "bands2.txt"),
        ]
        self.latest_spin_band_kpt = os.path.join(self.latest_band_spin_dir, "KPT_BANDS")
        self.latest_spin_band_stru = os.path.join(self.latest_band_spin_dir, "STRU")
        self.latest_spin_band_log = os.path.join(
            self.latest_band_spin_dir, "running_nscf.log"
        )

        self.lts_spin_band_files = [
            os.path.join(self.lts_band_spin_dir, "BANDS_1.dat"),
            os.path.join(self.lts_band_spin_dir, "BANDS_2.dat"),
        ]
        self.lts_spin_band_kpt = os.path.join(self.lts_band_spin_dir, "KPT_BANDS")
        self.lts_spin_band_stru = os.path.join(self.lts_band_spin_dir, "STRU")
        self.lts_spin_band_log = os.path.join(self.lts_band_spin_dir, "running_nscf.log")

        self.dos_file = os.path.join(self.latest_dos_dir, "doss1g1_nao.txt")
        self.latest_tdos_file = os.path.join(self.latest_dos_dir, "TDOS.dat")
        self.pdos_file = os.path.join(self.latest_dos_dir, "PDOS.dat")
        self.dos_stru = os.path.join(self.latest_dos_dir, "STRU")
        self.dos_log = os.path.join(self.latest_dos_dir, "running_nscf.log")

        self.lts_dos_file = os.path.join(self.lts_dos_dir, "DOS1_smearing.dat")
        self.lts_pdos_file = os.path.join(self.lts_dos_dir, "PDOS")
        self.lts_tdos_file = os.path.join(self.lts_dos_dir, "TDOS")
        self.lts_dos_stru = os.path.join(self.lts_dos_dir, "STRU")
        self.lts_dos_log = os.path.join(self.lts_dos_dir, "running_nscf.log")

        self.latest_spin_dos_files = [
            os.path.join(self.latest_dos_spin_dir, "doss1g1_nao.txt"),
            os.path.join(self.latest_dos_spin_dir, "doss2g1_nao.txt"),
        ]
        self.latest_spin_pdos_file = os.path.join(self.latest_dos_spin_dir, "PDOS.dat")
        self.latest_spin_dos_stru = os.path.join(self.latest_dos_spin_dir, "STRU")
        self.latest_spin_dos_log = os.path.join(
            self.latest_dos_spin_dir, "running_nscf.log"
        )

        self.lts_spin_dos_files = [
            os.path.join(self.lts_dos_spin_dir, "DOS1_smearing.dat"),
            os.path.join(self.lts_dos_spin_dir, "DOS2_smearing.dat"),
        ]
        self.lts_spin_pdos_file = os.path.join(self.lts_dos_spin_dir, "PDOS")
        self.lts_spin_dos_stru = os.path.join(self.lts_dos_spin_dir, "STRU")
        self.lts_spin_dos_log = os.path.join(self.lts_dos_spin_dir, "running_nscf.log")

    def test_read_structure(self):
        for stru_file in (self.band_stru, self.lts_band_stru):
            with self.subTest(stru_file=stru_file):
                structure = read_structure(stru_file)
                assert_allclose(
                    structure.lattice.matrix,
                    [
                        [0.0, 2.69880478, 2.69880478],
                        [2.69880478, 0.0, 2.69880478],
                        [2.69880478, 2.69880478, 0.0],
                    ],
                    atol=1e-5,
                )
                self.assertEqual(structure[0].specie.symbol, "Si")
                self.assertEqual(len(structure), 2)

    def test_read_structure_cartesian_variants(self):
        template = """ATOMIC_SPECIES
Si 28.085 Si.upf

LATTICE_CONSTANT
2.0

LATTICE_VECTORS
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

ATOMIC_POSITIONS
{coord_type}
Si
0.0
1
1.0 2.0 3.0
"""
        expected = {
            "Cartesian": [1.05835442, 2.11670884, 3.17506326],
            "Cartesian_au": [0.52917721, 1.05835442, 1.58753163],
            "Cartesian_angstrom": [1.0, 2.0, 3.0],
            "Cartesian_angstrom_center_xy": [1.52917721, 2.52917721, 3.0],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            for coord_type, expected_cart in expected.items():
                with self.subTest(coord_type=coord_type):
                    stru_file = os.path.join(tmpdir, f"{coord_type}.STRU")
                    with open(stru_file, "w") as handle:
                        handle.write(template.format(coord_type=coord_type))

                    structure = read_structure(stru_file)
                    assert_allclose(structure.cart_coords[0], expected_cart, atol=1e-6)

    def test_read_kpoint_labels(self):
        for kpt_file in (self.band_kpt, self.lts_band_kpt):
            with self.subTest(kpt_file=kpt_file):
                kpoints, labels = read_kpoint_labels(kpt_file)
                self.assertEqual(len(kpoints), 217)
                self.assertEqual(labels[r"\Gamma"], (0.0, 0.0, 0.0))
                self.assertEqual(labels["L"], (0.5, 0.5, 0.5))
                self.assertEqual(labels["W"], (0.5, 0.25, 0.75))
                self.assertEqual(labels["X"], (0.5, 0.0, 0.5))

    def test_read_kpoint_labels_line_mode_does_not_close_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kpt_file = os.path.join(tmpdir, "KPT")
            with open(kpt_file, "w") as handle:
                handle.write("K_POINTS\n")
                handle.write("3\n")
                handle.write("Line\n")
                handle.write("0.0 0.0 0.0 2 # Gamma\n")
                handle.write("0.5 0.0 0.0 2 # X\n")
                handle.write("0.5 0.5 0.0 7 # M\n")

            kpoints, labels = read_kpoint_labels(kpt_file)

            assert_allclose(
                kpoints,
                [
                    [0.0, 0.0, 0.0],
                    [0.25, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.25, 0.0],
                    [0.5, 0.5, 0.0],
                ],
            )
            self.assertEqual(labels[r"\Gamma"], (0.0, 0.0, 0.0))
            self.assertEqual(labels["X"], (0.5, 0.0, 0.0))
            self.assertEqual(labels["M"], (0.5, 0.5, 0.0))

    def test_read_efermi(self):
        self.assertAlmostEqual(read_efermi(self.band_log), 7.0139922244)
        self.assertAlmostEqual(read_efermi(self.lts_band_log), 7.0139922244)
        self.assertAlmostEqual(read_efermi(self.dos_log), 6.9837298674)
        self.assertAlmostEqual(read_efermi(self.lts_dos_log), 6.9837298674)

    def test_band_structure(self):
        bs = band_structure(
            self.band_text_file,
            kpt_file=self.band_kpt,
            stru_file=self.band_stru,
            log_file=self.band_log,
        )

        self.assertEqual(bs.bands[Spin.up].shape, (8, 217))
        self.assertFalse(bs.is_metal())
        self.assertLess(abs(bs.efermi - bs.get_vbm()["energy"]), 0.02)
        self.assertEqual(bs.kpoints[0].label, r"\Gamma")
        self.assertAlmostEqual(bs.bands[Spin.up][0, 0], -5.33891831)

    def test_band_structure_legacy_filename(self):
        bs = band_structure(
            self.lts_band_file,
            kpt_file=self.lts_band_kpt,
            stru_file=self.lts_band_stru,
            log_file=self.lts_band_log,
        )
        self.assertEqual(bs.bands[Spin.up].shape, (8, 217))

    def test_band_structure_spin_latest_and_lts(self):
        cases = (
            (
                self.latest_spin_band_files,
                self.latest_spin_band_kpt,
                self.latest_spin_band_stru,
                self.latest_spin_band_log,
            ),
            (
                self.lts_spin_band_files,
                self.lts_spin_band_kpt,
                self.lts_spin_band_stru,
                self.lts_spin_band_log,
            ),
        )
        for bands_files, kpt_file, stru_file, log_file in cases:
            with self.subTest(bands_files=bands_files):
                bs = band_structure(
                    bands_files,
                    kpt_file=kpt_file,
                    stru_file=stru_file,
                    log_file=log_file,
                )
                self.assertEqual(bs.bands[Spin.up].shape, (8, 217))
                self.assertIn(Spin.down, bs.bands)

    def test_read_tdos(self):
        dos, pdos = read_dos(self.dos_file, log_file=self.dos_log)

        self.assertFalse(pdos)
        self.assertEqual(len(dos.energies), 2295)
        self.assertIn(Spin.up, dos.densities)
        self.assertAlmostEqual(dos.efermi, 6.923347327864863)
        self.assertLess(dos.energies[0], dos.energies[-1])
        self.assertGreater(np.max(dos.densities[Spin.up]), 0)

    def test_read_tdos_lts_smearing_file(self):
        dos, pdos = read_dos(self.lts_dos_file, log_file=self.lts_dos_log)

        self.assertFalse(pdos)
        self.assertEqual(len(dos.energies), 2294)
        self.assertIn(Spin.up, dos.densities)
        self.assertAlmostEqual(dos.efermi, 6.923338509559195)
        self.assertLess(dos.energies[0], dos.energies[-1])
        self.assertGreater(np.max(dos.densities[Spin.up]), 0)

    def test_read_real_tdos_fixtures_from_latest_and_lts(self):
        cases = (
            (self.latest_tdos_file, self.dos_log),
            (self.lts_tdos_file, self.lts_dos_log),
        )
        for tdos_file, log_file in cases:
            with self.subTest(tdos_file=tdos_file):
                dos, pdos = read_dos(tdos_file, log_file=log_file)
                self.assertFalse(pdos)
                self.assertEqual(len(dos.energies), 2294)
                self.assertIn(Spin.up, dos.densities)

    def test_read_tdos_dos_txt_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dos_txt = os.path.join(tmpdir, "dos.txt")
            with open(dos_txt, "w") as handle:
                handle.write("1    # ionic step\n")
                handle.write("3 # number of points\n")
                handle.write(
                    "#        energy    elec_states     sum_states   states_smear     sum_states\n"
                )
                handle.write("       -5.0              0              0            1.25           0.01\n")
                handle.write("       -4.0              0              0            2.50           0.03\n")
                handle.write("       -3.0              0              0            3.75           0.06\n")

            dos, pdos = read_dos(dos_txt, log_file=self.dos_log)

            self.assertFalse(pdos)
            assert_allclose(dos.energies, [-5.0, -4.0, -3.0])
            assert_allclose(dos.densities[Spin.up], [1.25, 2.50, 3.75])
            self.assertAlmostEqual(dos.efermi, read_efermi(self.dos_log))

    def test_read_tdos_uses_final_complete_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dos_txt = os.path.join(tmpdir, "dos.txt")
            with open(dos_txt, "w") as handle:
                handle.write("1    # ionic step\n")
                handle.write("2 # number of points\n")
                handle.write("-5.0 0 0 1.25 0.01\n")
                handle.write("-4.0 0 0 2.50 0.03\n")
                handle.write("2    # ionic step\n")
                handle.write("2 # number of points\n")
                handle.write("-5.0 0 0 9.25 0.01\n")
                handle.write("-4.0 0 0 8.50 0.03\n")

            dos, pdos = read_dos(dos_txt, log_file=self.dos_log)

            self.assertFalse(pdos)
            assert_allclose(dos.energies, [-5.0, -4.0])
            assert_allclose(dos.densities[Spin.up], [9.25, 8.50])

    def test_read_tdos_tdos_dat_multispin_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tdos_dat = os.path.join(tmpdir, "TDOS.dat")
            with open(tdos_dat, "w") as handle:
                handle.write("-5.0 1.25 0.75\n")
                handle.write("-4.0 2.50 1.50\n")
                handle.write("-3.0 3.75 2.25\n")

            dos, pdos = read_dos(tdos_dat, log_file=self.dos_log)

            self.assertFalse(pdos)
            assert_allclose(dos.energies, [-5.0, -4.0, -3.0])
            assert_allclose(dos.densities[Spin.up], [1.25, 2.50, 3.75])
            assert_allclose(dos.densities[Spin.down], [0.75, 1.50, 2.25])
            self.assertAlmostEqual(dos.efermi, read_efermi(self.dos_log))

    def test_read_tdos_tdos_legacy_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tdos = os.path.join(tmpdir, "TDOS")
            with open(tdos, "w") as handle:
                handle.write("-5.0 1.25 0.75\n")
                handle.write("-4.0 2.50 1.50\n")
                handle.write("-3.0 3.75 2.25\n")

            dos, pdos = read_dos(tdos, log_file=self.dos_log)

            self.assertFalse(pdos)
            assert_allclose(dos.energies, [-5.0, -4.0, -3.0])
            assert_allclose(dos.densities[Spin.up], [1.25, 2.50, 3.75])
            assert_allclose(dos.densities[Spin.down], [0.75, 1.50, 2.25])
            self.assertAlmostEqual(dos.efermi, read_efermi(self.dos_log))

    def test_find_abacus_dos_files_prefers_new_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "OUT.ABACUS")
            os.makedirs(out_dir)
            open(os.path.join(out_dir, "TDOS.dat"), "w").close()
            open(os.path.join(out_dir, "doss1g1_nao.txt"), "w").close()
            open(os.path.join(out_dir, "doss2g1_nao.txt"), "w").close()

            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                tdos_files = _find_abacus_dos_files()
            finally:
                os.chdir(cwd)

            self.assertEqual(
                tdos_files,
                [
                    os.path.join("OUT.ABACUS", "doss1g1_nao.txt"),
                    os.path.join("OUT.ABACUS", "doss2g1_nao.txt"),
                ],
            )

    def test_find_abacus_dos_files_accepts_tdos_without_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "OUT.ABACUS")
            os.makedirs(out_dir)
            open(os.path.join(out_dir, "TDOS"), "w").close()

            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                tdos_files = _find_abacus_dos_files()
            finally:
                os.chdir(cwd)

            self.assertEqual(tdos_files, [os.path.join("OUT.ABACUS", "TDOS")])

    def test_abacus_band_partner_lookup_accepts_second_spin_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            band_1 = os.path.join(tmpdir, "bands1.txt")
            band_2 = os.path.join(tmpdir, "bands2.txt")
            open(band_1, "w").close()
            open(band_2, "w").close()

            self.assertEqual(
                _with_abacus_band_partner(band_2),
                [band_1, band_2],
            )

    def test_abacus_dos_partner_lookup_accepts_second_spin_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dos_1 = os.path.join(tmpdir, "doss1g1_nao.txt")
            dos_2 = os.path.join(tmpdir, "doss2g1_nao.txt")
            open(dos_1, "w").close()
            open(dos_2, "w").close()

            self.assertEqual(
                _with_abacus_spin_partner(dos_2),
                [dos_1, dos_2],
            )

    def test_abacus_dos_partner_lookup_accepts_second_legacy_spin_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dos_1 = os.path.join(tmpdir, "DOS1_smearing.dat")
            dos_2 = os.path.join(tmpdir, "DOS2_smearing.dat")
            open(dos_1, "w").close()
            open(dos_2, "w").close()

            self.assertEqual(
                _with_abacus_spin_partner(dos_2),
                [dos_1, dos_2],
            )

    def test_read_pdos(self):
        cases = (
            (self.dos_file, self.pdos_file, self.dos_stru, self.dos_log),
            (
                self.lts_dos_file,
                self.lts_pdos_file,
                self.lts_dos_stru,
                self.lts_dos_log,
            ),
        )
        for tdos_file, pdos_file, stru_file, log_file in cases:
            with self.subTest(tdos_file=tdos_file):
                dos, pdos = read_dos(
                    tdos_file,
                    pdos_file=pdos_file,
                    stru_file=stru_file,
                    log_file=log_file,
                )

                self.assertEqual(set(pdos.keys()), {"Si"})
                self.assertIn("s", pdos["Si"])
                self.assertIn("p", pdos["Si"])
                self.assertIn("d", pdos["Si"])

                projected_sum = sum(
                    orbital_dos.densities[Spin.up] for orbital_dos in pdos["Si"].values()
                )
                assert_allclose(projected_sum, dos.densities[Spin.up], atol=1e-5)

    def test_read_spin_dos_latest_and_lts(self):
        cases = (
            (
                self.latest_spin_dos_files,
                self.latest_spin_pdos_file,
                self.latest_spin_dos_stru,
                self.latest_spin_dos_log,
            ),
            (
                self.lts_spin_dos_files,
                self.lts_spin_pdos_file,
                self.lts_spin_dos_stru,
                self.lts_spin_dos_log,
            ),
        )
        for tdos_files, pdos_file, stru_file, log_file in cases:
            with self.subTest(tdos_files=tdos_files):
                dos, pdos = read_dos(
                    tdos_files,
                    pdos_file=pdos_file,
                    stru_file=stru_file,
                    log_file=log_file,
                )
                self.assertIn(Spin.down, dos.densities)
                self.assertEqual(set(pdos.keys()), {"Si"})

    def test_read_spin_dos_lts_is_order_independent(self):
        dos_forward, _ = read_dos(self.lts_spin_dos_files, log_file=self.lts_spin_dos_log)
        dos_reverse, _ = read_dos(
            list(reversed(self.lts_spin_dos_files)),
            log_file=self.lts_spin_dos_log,
        )

        assert_allclose(dos_forward.energies, dos_reverse.energies)
        assert_allclose(dos_forward.densities[Spin.up], dos_reverse.densities[Spin.up])
        assert_allclose(dos_forward.densities[Spin.down], dos_reverse.densities[Spin.down])

    def test_write_kpoints_bands_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kpoints = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])
            labels = [r"@\Gamma", "", "X"]

            write_kpoint_files(self.band_stru, kpoints, labels, directory=tmpdir)

            kpt_file = os.path.join(tmpdir, "KPT_BANDS")
            with open(kpt_file) as handle:
                lines = handle.readlines()

            self.assertEqual(lines[2].strip(), "Direct")
            self.assertIn("# @Gamma", lines[3])
            self.assertIn("# X", lines[5])

            parsed_kpoints, parsed_labels = read_kpoint_labels(kpt_file)
            assert_allclose(parsed_kpoints, kpoints)
            self.assertEqual(parsed_labels[r"@\Gamma"], (0.0, 0.0, 0.0))
            self.assertEqual(parsed_labels["X"], (0.5, 0.0, 0.0))

    def test_kgen_abacus_writes_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                kgen(filename=self.band_stru, code="abacus", density=5)
            finally:
                os.chdir(cwd)

            kpt_file = os.path.join(tmpdir, "KPT_BANDS")
            self.assertTrue(os.path.exists(kpt_file))

            with open(kpt_file) as handle:
                lines = handle.readlines()

            self.assertEqual(lines[0].strip(), "K_POINTS")
            self.assertEqual(lines[2].strip(), "Direct")
            self.assertTrue(any("#" in line for line in lines[3:]))

            kpoints, labels = read_kpoint_labels(kpt_file)
            self.assertGreater(len(kpoints), 0)
            self.assertIn(r"\Gamma", labels)

    def test_bandplot_cli_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            band_file = os.path.join(tmpdir, "band.txt")
            stru_file = os.path.join(tmpdir, "STRU")
            log_file = os.path.join(tmpdir, "running_nscf.log")
            kpt_file = os.path.join(tmpdir, "KPT_BANDS")
            shutil.copyfile(self.band_text_file, band_file)
            shutil.copyfile(self.band_stru, stru_file)
            shutil.copyfile(self.band_log, log_file)
            shutil.copyfile(self.band_kpt, kpt_file)

            bandplot(
                filenames=[band_file],
                code="abacus",
                dos_file=self.dos_file,
                directory=tmpdir,
            )
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "band.pdf")))

    def test_bandplot_rejects_projected_abacus(self):
        with self.assertRaises(NotImplementedError):
            bandplot(
                filenames=[self.band_text_file],
                code="abacus",
                projection_selection=[("Si", ("s",))],
                plt=plt,
            )

    def test_dosplot_cli_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dosplot(
                filename=self.dos_file,
                code="abacus",
                directory=tmpdir,
            )
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "dos.pdf")))
