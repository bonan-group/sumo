import logging
import os
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos

from sumo.electronic_structure.dos import get_pdos

_bohr_to_angstrom = 0.5291772108

logger = logging.getLogger("io.abacus")


def read_structure(stru_file):
    """Read an ABACUS STRU file into a pymatgen Structure."""

    blocks = _read_stru_blocks(stru_file)

    if "lattice_vectors" not in blocks or "lattice_constant" not in blocks:
        raise OSError(f"Could not find lattice information in {stru_file}")

    lattice_constant = float(_strip_comment(blocks["lattice_constant"][0]))
    lattice_vectors = np.array(
        [list(map(float, _strip_comment(line).split())) for line in blocks["lattice_vectors"]],
        dtype=float,
    )
    lattice = Lattice(lattice_vectors * lattice_constant * _bohr_to_angstrom)

    if "atomic_positions" not in blocks:
        raise OSError(f"Could not find atomic positions in {stru_file}")

    pos_lines = [_strip_comment(line) for line in blocks["atomic_positions"] if _strip_comment(line)]
    coord_type = pos_lines[0].lower()

    species = []
    coords = []
    i = 1
    while i < len(pos_lines):
        symbol = pos_lines[i]
        natom = int(float(pos_lines[i + 2]))
        atom_lines = pos_lines[i + 3 : i + 3 + natom]
        for line in atom_lines:
            species.append(symbol)
            coords.append(list(map(float, line.split()[:3])))
        i += 3 + natom

    if coord_type.startswith("direct"):
        return Structure(lattice, species, coords, coords_are_cartesian=False)
    if coord_type.startswith("cartesian"):
        return Structure(
            lattice,
            species,
            _cartesian_coords_from_stru(
                np.array(coords, dtype=float), coord_type, lattice, lattice_constant
            ),
            coords_are_cartesian=True,
        )

    raise NotImplementedError(
        f"Unsupported ABACUS coordinate mode '{pos_lines[0]}' in {stru_file}"
    )


def read_kpoint_labels(kpt_file):
    """Read ABACUS KPT files, including explicit KPT_BANDS/KPT and line-mode inputs."""

    with open(kpt_file) as handle:
        raw_lines = handle.readlines()

    lines = [_strip_comment(line) for line in raw_lines if _strip_comment(line)]
    if not lines or lines[0].upper() not in {"K_POINTS", "KPOINTS", "K"}:
        raise OSError(f"{kpt_file} is not an ABACUS KPT file")

    nentries = int(lines[1])
    mode = lines[2].strip().lower()
    if mode == "line":
        special_k = []
        numbers = []
        labels = []
        raw_idx = 0
        for line in raw_lines:
            clean = _strip_comment(line)
            if not clean:
                continue
            raw_idx += 1
            if raw_idx <= 3:
                continue
            tokens = clean.split()
            if len(tokens) < 4:
                continue
            special_k.append(list(map(float, tokens[:3])))
            numbers.append(int(tokens[3]))

            label = ""
            if "#" in line:
                label = line.split("#", 1)[1].strip()
            label = _normalise_label(label)
            labels.append(label)
            if len(special_k) == nentries:
                break

        if len(special_k) != nentries:
            raise OSError(f"Expected {nentries} special k-points in {kpt_file}")

        return _full_kpath(special_k, numbers), {
            label: tuple(coords)
            for label, coords in zip(labels, special_k)
            if label
        }

    if mode == "direct":
        kpoints = []
        labels = {}
        raw_idx = 0
        for line in raw_lines:
            clean = _strip_comment(line)
            if not clean:
                continue
            raw_idx += 1
            if raw_idx <= 3:
                continue
            tokens = clean.split()
            if len(tokens) < 4:
                continue
            coords = tuple(map(float, tokens[:3]))
            kpoints.append(coords)

            if "#" in line:
                label = _normalise_label(line.split("#", 1)[1].strip())
                if label:
                    labels[label] = coords
            if len(kpoints) == nentries:
                break

        if len(kpoints) != nentries:
            raise OSError(f"Expected {nentries} explicit k-points in {kpt_file}")

        return np.array(kpoints), labels

    raise NotImplementedError(
        f"ABACUS band plotting currently supports Direct and Line KPT modes, found '{lines[2]}'"
    )


def write_kpoint_files(
    filename,
    kpoints,
    labels,
    make_folders=False,
    ibzkpt=None,
    kpts_per_split=None,
    directory=None,
    cart_coords=False,
):
    """Write ABACUS explicit k-point paths in Direct coordinates."""

    if make_folders:
        logging.info(
            'Ignoring ABACUS k-point write option "make_folders"; not implemented.'
        )
    if ibzkpt is not None:
        logging.info(
            'Ignoring ABACUS k-point write option "ibzkpt"; not implemented.'
        )
    if kpts_per_split is not None:
        logging.info(
            'Ignoring ABACUS k-point write option "kpts_per_split"; not implemented.'
        )
    if cart_coords:
        logging.warning(
            "Ignoring request for Cartesian coordinates: ABACUS KPT_BANDS is written in Direct coordinates."
        )

    path = directory if directory is not None else os.path.curdir
    os.makedirs(path, exist_ok=True)
    output_file = os.path.join(path, "KPT_BANDS")

    with open(output_file, "w") as handle:
        handle.write("K_POINTS\n")
        handle.write(f"{len(kpoints)}\n")
        handle.write("Direct\n")
        for kpoint, label in zip(kpoints, labels):
            line = f"{kpoint[0]:11.8f} {kpoint[1]:11.8f} {kpoint[2]:11.8f} 1.0"
            if label:
                line += f" # {_abacus_comment_label(label)}"
            handle.write(line + "\n")


def read_efermi(log_file):
    """Read the final ABACUS EFERMI value in eV."""

    efermi = None
    re_explicit = re.compile(r"EFERMI\s*=\s*([+-]?\d+(?:\.\d+)?)\s*eV", re.I)
    re_iter = re.compile(r"E_Fermi\s+[+-]?\d+(?:\.\d+)?\s+([+-]?\d+(?:\.\d+)?)")

    with open(log_file) as handle:
        for line in handle:
            match = re_explicit.search(line)
            if match:
                efermi = float(match.group(1))
                continue
            match = re_iter.search(line)
            if match:
                efermi = float(match.group(1))

    if efermi is None:
        raise OSError(f"Could not find EFERMI in {log_file}")
    return efermi


def band_structure(bands_files, kpt_file=None, stru_file=None, log_file=None):
    """Convert ABACUS band.txt, bands*.txt, or BANDS_*.dat output to a BandStructureSymmLine."""

    if isinstance(bands_files, str):
        bands_files = [bands_files]

    if len(bands_files) not in (1, 2):
        raise ValueError("ABACUS band structure expects one or two ABACUS band output files")

    if not kpt_file:
        raise OSError("ABACUS band plotting requires a KPT_BANDS or explicit KPT file")
    if not stru_file:
        raise OSError("ABACUS band plotting requires a STRU file")

    structure = read_structure(stru_file)
    kpoints, labels = read_kpoint_labels(kpt_file)

    bands = {}
    for spin, filename in zip(_spin_channels(len(bands_files)), _sort_abacus_outputs(bands_files)):
        data = np.loadtxt(filename)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data = _deduplicate_band_rows(data, len(kpoints), filename)
        eigs = data[:, 2:]
        if len(kpoints) != eigs.shape[0]:
            raise ValueError(
                f"K-point count mismatch between {kpt_file} ({len(kpoints)}) and {filename} ({eigs.shape[0]})"
            )
        bands[spin] = eigs.T

    efermi = read_efermi(log_file) if log_file else 0.0
    if log_file is None:
        logger.warning("No ABACUS log file found; using 0 eV as the Fermi reference.")

    if not _is_metal(bands, efermi):
        efermi = _get_vbm(bands, efermi)

    return BandStructureSymmLine(
        kpoints,
        bands,
        structure.lattice.reciprocal_lattice_crystallographic,
        efermi,
        labels,
        coords_are_cartesian=False,
        structure=structure,
    )


def read_dos(
    tdos_files=None,
    pdos_file=None,
    stru_file=None,
    log_file=None,
    gaussian=None,
    lm_orbitals=None,
    elements=None,
    atoms=None,
    total_only=False,
):
    """Convert ABACUS DOS output to Sumo/pymatgen DOS objects."""

    total_dos = _read_tdos(tdos_files, log_file) if tdos_files else None
    complete_dos = None
    if pdos_file and not total_only:
        if not stru_file:
            raise OSError("ABACUS PDOS plotting requires a STRU file")
        complete_dos = _read_pdos(pdos_file, stru_file, total_dos, gaussian=gaussian)
        total_dos = Dos(
            complete_dos.efermi,
            complete_dos.energies,
            complete_dos.densities,
        )

    if total_dos is None:
        raise OSError(
            "Need ABACUS DOS data: provide doss*.txt, TDOS/TDOS.dat, DOS*_smearing.dat, dos.txt, or PDOS"
        )

    if gaussian:
        total_dos.densities = total_dos.get_smeared_densities(gaussian)
        if complete_dos is not None:
            complete_dos.densities = complete_dos.get_smeared_densities(gaussian)
            for site, orbitals in complete_dos.pdos.items():
                for orbital in orbitals:
                    complete_dos.pdos[site][orbital] = complete_dos.get_site_orbital_dos(
                        site, orbital
                    ).get_smeared_densities(gaussian)

    if complete_dos is None or total_only:
        return _adjust_dos_reference(total_dos), {}

    complete_dos = _adjust_complete_dos_reference(complete_dos)
    pdos = get_pdos(
        complete_dos,
        lm_orbitals=lm_orbitals,
        atoms=atoms,
        elements=elements,
    )
    return complete_dos, pdos


def _read_tdos(tdos_files, log_file=None):
    if isinstance(tdos_files, str):
        tdos_files = [tdos_files]

    if len(tdos_files) not in (1, 2):
        raise ValueError("ABACUS DOS expects one or two total-DOS files")

    if len(tdos_files) == 1:
        energies, densities = _read_single_tdos_file(tdos_files[0])
    else:
        densities = {}
        energies = None
        for spin, filename in zip(_spin_channels(len(tdos_files)), _sort_abacus_outputs(tdos_files)):
            energies_i, density_i = _read_single_tdos_channel(filename)
            if energies is None:
                energies = energies_i
            elif not np.allclose(energies, energies_i, atol=1e-5):
                raise ValueError(f"Energy grid mismatch in {filename}")
            densities[spin] = density_i

    efermi = read_efermi(log_file) if log_file else 0.0
    if log_file is None:
        logger.warning("No ABACUS log file found; using 0 eV as the Fermi reference.")
    return Dos(efermi, energies, densities)


def _read_single_tdos_file(filename):
    data = _read_tdos_table(filename)

    if _is_multispin_tdos(filename):
        densities = {}
        channels = data[:, 1:]
        if channels.shape[1] == 1:
            densities[Spin.up] = channels[:, 0]
        elif channels.shape[1] == 2:
            densities[Spin.up] = channels[:, 0]
            densities[Spin.down] = channels[:, 1]
        else:
            raise ValueError(
                f"Unsupported number of spin channels in {filename}: found {channels.shape[1]}"
            )
        return data[:, 0], densities

    energies, density = _read_single_tdos_channel(filename)
    return energies, {Spin.up: density}


def _read_single_tdos_channel(filename):
    data = _read_tdos_table(filename)
    dos_column = 3 if data.shape[1] >= 5 else 1
    return data[:, 0], data[:, dos_column]


def _read_tdos_table(filename):
    rows = []
    blocks = []
    expected_points = None
    current_block = []
    with open(filename) as handle:
        for line in handle:
            if "number of points" in line.lower():
                try:
                    expected_points = int(float(_strip_comment(line)))
                    current_block = []
                except ValueError:
                    expected_points = None
                continue

            clean = _strip_comment(line)
            if not clean:
                continue
            try:
                values = [float(value) for value in clean.split()]
            except ValueError:
                continue
            if len(values) < 2:
                continue
            rows.append(values)

            if expected_points is not None:
                current_block.append(values)
                if len(current_block) == expected_points:
                    blocks.append(current_block)
                    expected_points = None
                    current_block = []

    if blocks:
        if len(blocks) > 1:
            logger.warning(
                "Detected %d DOS blocks in %s; using the final complete block.",
                len(blocks),
                filename,
            )
        return np.array(blocks[-1], dtype=float)

    if not rows:
        raise ValueError(f"Could not parse ABACUS DOS data from {filename}")

    return np.array(rows, dtype=float)


def _is_multispin_tdos(filename):
    return os.path.basename(filename) in {"TDOS", "TDOS.dat"}


def _read_pdos(pdos_file, stru_file, total_dos=None, gaussian=None):
    root = ET.parse(pdos_file).getroot()

    nspin = int(root.findtext("nspin", default="1").strip())
    energies = np.array(
        [
            float(value)
            for value in (root.findtext("energy_values", default="") or "").split()
        ],
        dtype=float,
    )
    structure = read_structure(stru_file)

    pdoss = OrderedDict((site, {}) for site in structure.sites)
    total_from_pdos = {spin: np.zeros(len(energies)) for spin in _spin_channels(nspin)}

    for orbital_xml in root.findall("orbital"):
        atom_index = int(orbital_xml.attrib["atom_index"]) - 1
        l_index = int(orbital_xml.attrib["l"])
        m_index = int(orbital_xml.attrib["m"])
        orbital = _orbital_from_lm(l_index, m_index)
        if orbital is None:
            logger.warning(
                "Skipping unsupported ABACUS PDOS channel l=%s, m=%s in %s",
                l_index,
                m_index,
                pdos_file,
            )
            continue

        data = np.array(
            [[float(x) for x in line.split()] for line in _non_empty_lines(orbital_xml.findtext("data", default=""))],
            dtype=float,
        )

        spin_densities = {}
        for spin_index, spin in enumerate(_spin_channels(nspin)):
            spin_densities[spin] = data[:, spin_index]
            total_from_pdos[spin] += data[:, spin_index]

        site = structure.sites[atom_index]
        if orbital in pdoss[site]:
            for spin in spin_densities:
                pdoss[site][orbital][spin] += spin_densities[spin]
        else:
            pdoss[site][orbital] = spin_densities

    if total_dos is None:
        efermi = 0.0
        total_dos = Dos(efermi, energies, total_from_pdos)
    else:
        if len(total_dos.energies) != len(energies) or not np.allclose(
            total_dos.energies, energies, atol=1e-5
        ):
            logger.warning(
                "ABACUS DOS and PDOS energy grids do not match in %s; using the PDOS energy grid.",
                pdos_file,
            )
            total_dos = Dos(total_dos.efermi, energies, total_from_pdos)

    return CompleteDos(structure, total_dos, pdoss)


def _adjust_dos_reference(dos):
    gap = dos.get_gap()
    if gap > 0:
        _, vbm = dos.get_cbm_vbm()
        dos.efermi = vbm
    return dos


def _adjust_complete_dos_reference(complete_dos):
    _adjust_dos_reference(complete_dos)
    return complete_dos


def _orbital_from_lm(l_index, m_index):
    mapping = {
        0: {0: Orbital.s},
        1: {0: Orbital.pz, 1: Orbital.px, 2: Orbital.py},
        2: {
            0: Orbital.dz2,
            1: Orbital.dxz,
            2: Orbital.dyz,
            3: Orbital.dx2,
            4: Orbital.dxy,
        },
        3: {
            0: Orbital.f0,
            1: Orbital.f1,
            2: Orbital.f2,
            3: Orbital.f3,
            4: Orbital.f_1,
            5: Orbital.f_2,
            6: Orbital.f_3,
        },
    }
    return mapping.get(l_index, {}).get(m_index)


def _spin_channels(nspin):
    return (Spin.up,) if nspin == 1 else (Spin.up, Spin.down)


def _sort_abacus_outputs(filenames):
    def _key(filename):
        basename = os.path.basename(filename)
        patterns = (
            r"[Bb][Aa][Nn][Dd][Ss](\d+)(?:\.\w+)?$",
            r"[Dd][Oo][Ss][Ss](\d+)",
            r"[Dd][Oo][Ss](\d+)_smearing(?:\.\w+)?$",
            r"_(\d+)(?:\.\w+)?$",
        )
        for pattern in patterns:
            match = re.search(pattern, basename)
            if match:
                return int(match.group(1))
        return 1

    return sorted(filenames, key=_key)


def _read_stru_blocks(filename):
    block_names = {
        "ATOMIC_SPECIES",
        "NUMERICAL_ORBITAL",
        "NUMERICAL_DESCRIPTOR",
        "LATTICE_CONSTANT",
        "LATTICE_PARAMETER",
        "LATTICE_VECTORS",
        "ATOMIC_POSITIONS",
    }

    with open(filename) as handle:
        lines = [line.rstrip("\n") for line in handle]

    blocks = OrderedDict()
    current = None
    for raw_line in lines:
        clean = _strip_comment(raw_line)
        if not clean:
            continue
        if clean.upper() in block_names:
            current = clean.lower()
            blocks[current] = []
            continue
        if current is not None:
            blocks[current].append(raw_line)
    return blocks


def _strip_comment(line):
    return line.split("#", 1)[0].split("//", 1)[0].strip()


def _non_empty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def _normalise_label(label):
    hidden = label.startswith("@")
    core = label[1:] if hidden else label
    if core.lower() in {"g", "gamma", "\\gamma"}:
        core = r"\Gamma"
    return f"@{core}" if hidden else core


def _abacus_comment_label(label):
    hidden = label.startswith("@")
    core = label[1:] if hidden else label
    if core == r"\Gamma":
        core = "Gamma"
    return f"@{core}" if hidden else core


def _deduplicate_band_rows(data, nkpoints, filename):
    if data.shape[0] == nkpoints:
        return data

    if nkpoints and data.shape[0] % nkpoints == 0:
        nblocks = data.shape[0] // nkpoints
        blocks = [data[i * nkpoints : (i + 1) * nkpoints] for i in range(nblocks)]
        if all(np.allclose(blocks[0], block) for block in blocks[1:]):
            logger.info(
                "Detected %d repeated k-point blocks in %s; using the first block.",
                nblocks,
                filename,
            )
            return blocks[0]

    return data


def _full_kpath(special_k, numbers):
    special_k = np.array(special_k, dtype=float)
    numbers = np.array(numbers, dtype=int)
    if len(special_k) == 1:
        return special_k.copy()

    coords = []
    for start, end, npts in zip(special_k[:-1], special_k[1:], numbers[:-1]):
        segment = start + (end - start) * np.arange(npts).reshape(-1, 1) / npts
        coords.extend(segment.tolist())
    coords.append(special_k[-1].tolist())
    return np.array(coords, dtype=float)


def _cartesian_coords_from_stru(coords, coord_type, lattice, lattice_constant):
    if coord_type == "cartesian":
        return coords * lattice_constant * _bohr_to_angstrom
    if coord_type == "cartesian_au":
        return coords * _bohr_to_angstrom
    if coord_type == "cartesian_angstrom":
        return coords
    if coord_type.startswith("cartesian_angstrom_center_"):
        return coords + _cartesian_center_offset(coord_type, lattice)
    raise NotImplementedError(f"Unsupported ABACUS coordinate mode '{coord_type}'")


def _cartesian_center_offset(coord_type, lattice):
    centers = {
        "cartesian_angstrom_center_xy": (0.5, 0.5, 0.0),
        "cartesian_angstrom_center_xz": (0.5, 0.0, 0.5),
        "cartesian_angstrom_center_yz": (0.0, 0.5, 0.5),
        "cartesian_angstrom_center_xyz": (0.5, 0.5, 0.5),
    }
    if coord_type not in centers:
        raise NotImplementedError(f"Unsupported ABACUS coordinate mode '{coord_type}'")
    return np.dot(np.array(centers[coord_type], dtype=float), lattice.matrix)


def _is_metal(bands, efermi):
    for spin_bands in bands.values():
        for band in spin_bands:
            if np.any(band < efermi) and np.any(band > efermi):
                return True
    return False


def _get_vbm(bands, efermi):
    occupied = []
    for spin_bands in bands.values():
        for band in spin_bands:
            occupied.extend(band[band < efermi])
    return max(occupied) if occupied else efermi
