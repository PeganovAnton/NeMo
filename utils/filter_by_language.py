import argparse
import logging
import multiprocessing as mp
import re
import shutil
import warnings
from pathlib import Path

from langdetect import detect


logging.basicConfig(level=logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_src",
        help="Path to the input source file.",
        type=Path,
    )
    parser.add_argument(
        "--input-tgt",
        "-i",
        help="Path to the input target file.",
        type=Path,
    )
    parser.add_argument(
        "output_src",
        help="Path to the file which will contain output source.",
        type=Path,
    )
    parser.add_argument(
        "--output-tgt",
        "-o",
        help="Path to the output target file",
        type=Path,
    )
    parser.add_argument(
        "source_lang",
        help="Input language. See https://github.com/Mimino666/langdetect."
    )
    parser.add_argument(
        "--target-lang",
        "-L",
        help="Output language. See https://github.com/Mimino666/langdetect."
    )
    parser.add_argument(
        "removed_src",
        help="Path to file where removed source lines will be saved",
        type=Path,
    )
    parser.add_argument(
        "--removed-tgt",
        "-r",
        help="Path to file where removed target lines will be saved",
        type=Path,
    )
    args = parser.parse_args()
    if not (args.output_tgt is None and args.input_tgt is None and args.source_lang is None and args.removed_src is None \
            or args.output_tgt is not None and args.input_tgt is not None and args.target_lang is not None and args.removed_tgt is not None):
        raise ValueError(
            f"Arguments `input_tgt`, `output_tgt`, `target_lang` have to be either `None` simultaneously or not `None`"
            f"simultaneously. Given input_tgt={args.input_tgt}, output_tgt={args.output_tgt}, "
            f"target_lang={args.target_lang}")
    args.input_src = args.input_src.expanduser()
    if args.input_tgt is not None:
        args.input_tgt = args.input_tgt.expanduser()
    args.output_src = args.output_src.expanduser()
    if args.output_tgt is not None:
        args.output_tgt = args.output_tgt.expanduser()
    args.removed_src = args.removed_src.expanduser()
    if args.removed_tgt is not None:
        args.removed_tgt = args.removed_tgt.expanduser()
    return args


def get_lang(line, fn):
    try:
       lang = detect(line)
    except:
       #warnings.warn(f"No features found in line {repr(line)} in file {fn}")
       lang = None
    return lang


def get_edges_in_1_file(fn, num_parts):
    num_lines = 0
    edges = [0]
    with open(fn) as f:
        while True:
            c = f.read(1)
            if not c:
                break
            if c == '\n':
                num_lines += 1
                edges.append(f.tell()+1)
        if edges[-1] < f.tell():
            edges.append(f.tell()+1)
            num_lines += 1
    return [edges[int(i*num_lines/num_parts)] for i in range(num_parts)] + [edges[-1]], num_lines


def get_edges(src_fn, tgt_fn, num_parts):
    src_edges, src_num_lines = get_edges_in_1_file(src_fn, num_parts)
    assert num_parts + 1 == len(src_edges)
    src_edges = [(src_edges[i], src_edges[i+1]) for i in range(len(src_edges)-1)]
    if tgt_fn is not None:
        tgt_edges, tgt_num_lines = get_edges_in_1_file(tgt_fn, num_parts)
        tgt_edges = [(tgt_edges[i], tgt_edges[i + 1]) for i in range(len(tgt_edges) - 1)]
        if tgt_num_lines != src_num_lines:
            raise ValueError(f"Source {repr(src_fn)} and target {repr(tgt_fn)} files have different number of lines "
                             f"{src_num_lines} and {tgt_num_lines} correspondingly.")

    else:
        tgt_edges = [None] * num_parts
    assert len(src_edges) == num_parts
    return src_edges, tgt_edges


def filter_by_lang(args):
    (
        src_edges,
        tgt_edges,
        input_src,
        input_tgt,
        filtered_dir_src,
        filtered_dir_tgt,
        removed_dir_src,
        removed_dir_tgt,
        source_lang,
        target_lang,
        rank,
    ) = args
    output_src = filtered_dir_src / Path(f"rank{rank}")
    output_src_removed = removed_dir_src / Path(f"rank{rank}")
    if input_tgt is None:
        if tgt_edges is not None:
            warnings.warn("If input target is not provided `tgt_edges` argument is expected to be `None`")
        with open(input_src) as in_f, open(output_src, 'w') as out_f, open(output_src_removed, 'w') as out_r_f:
            in_f.seek(src_edges[0])
            i, l = 0, in_f.readline()
            while l:
                l = l.strip()
                in_lang = get_lang(l, input_src)
                if in_lang is None or in_lang != source_lang:
                    out_r_f.write(l + '\n')
                else:
                    out_f.write(l + '\n')
                if in_f.tell() >= src_edges[1]:
                    break
                i, l = i+1, in_f.readline()
    else:
        output_tgt = filtered_dir_tgt / Path(f"rank{rank}")
        output_tgt_removed = removed_dir_tgt / Path(f"rank{rank}")
        with open(input_src) as in_src, open(input_tgt) as in_tgt, open(output_src, 'w') as out_src, \
                open(output_tgt, 'w') as out_tgt, open(output_src_removed, 'w') as out_r_src, \
                open(output_tgt_removed, 'w') as out_r_tgt:
            in_src.seek(src_edges[0])
            in_tgt.seek(tgt_edges[0])
            src_l, tgt_l, i = in_src.readline(), in_tgt.readline(), 0
            while src_l and tgt_l:
                src_l = src_l.strip()
                tgt_l = tgt_l.strip()
                src_lang = get_lang(src_l, input_src)
                if src_lang is not None:
                   tgt_lang = get_lang(tgt_l, input_tgt)
                if src_lang is None or tgt_lang is None or src_lang != source_lang or tgt_lang != target_lang:
                    out_r_src.write(src_l + '\n')
                    out_r_tgt.write(tgt_l + '\n')
                else:
                    out_src.write(src_l + '\n')
                    out_tgt.write(tgt_l + '\n')
                if in_src.tell() >= src_edges[1]:
                    if in_tgt.tell() < tgt_edges[1]:
                        raise ValueError(
                            f"Edges of target and source has to be reached simultaneously, whereas "
                            f"in_src.tell()={in_src.tell()}, in_tgt.tell()={in_tgt.tell()}, "
                            f"src_edges[1]={src_edges[1]}, tgt_edges[1]={tgt_edges[1]}."
                        )
                    break
                if in_tgt.tell() >= tgt_edges[1]:
                    raise ValueError(
                        f"Edges of target and source has to be reached simultaneously, whereas "
                        f"in_src.tell()={in_src.tell()}, in_tgt.tell()={in_tgt.tell()}, "
                        f"src_edges[1]={src_edges[1]}, tgt_edges[1]={tgt_edges[1]}."
                    )
                src_l, tgt_l, i = in_src.readline(), in_tgt.readline(), i + 1

def _cat_results(out_file, tmp_dir):
    file_name_pattern = re.compile(r"/rank[1-9][\d]*$")
    with out_file.open('w') as out_f:
        for f in tmp_dir.iterdir():
            if not f.is_file():
                warnings.warn(f"Unexpected not file {f}")
            elif not file_name_pattern.search(str(f)):
                warnings.warn(f"Unexpected file {f}")
            else:
                with f.open('r') as in_f:
                    for l in in_f:
                        out_f.write(l)


def cat_results(out_files, tmp_dirs):
    for o_f, t_d in zip(out_files, tmp_dirs):
        if o_f is None or t_d is None:
            if o_f is not None or t_d is not None:
                warnings.warn(
                    f"Both output file and tmp directory expected to be `None` whereas tmp directory is {t_d} "
                    f"and output file is {o_f}."
                )
        _cat_results(o_f, t_d)


def main():
    args = get_args()
    tmp_dir = Path("tmp")
    tmp_filtered = tmp_dir / Path("filtered")
    tmp_filtered_src = tmp_filtered / Path("src")
    tmp_filtered_src.mkdir(parents=True, exist_ok=True)
    if args.input_tgt is None:
        tmp_filtered_tgt = None
    else:
        tmp_filtered_tgt = tmp_filtered / Path("tgt")
        tmp_filtered_tgt.mkdir(parents=True, exist_ok=True)
    tmp_removed = tmp_dir / Path("removed")
    tmp_removed_src = tmp_removed / Path("src")
    tmp_removed_src.mkdir(parents=True, exist_ok=True)
    if args.input_tgt is None:
        tmp_removed_tgt = None
    else:
        tmp_removed_tgt = tmp_removed / Path("tgt")
        tmp_removed_tgt.mkdir(parents=True, exist_ok=True)
    num_cpu = mp.cpu_count()
    src_edges, tgt_edges = get_edges(args.input_src, args.input_tgt, num_cpu)
    with mp.Pool(num_cpu) as pool:
        pool.map(
            filter_by_lang,
            [
                (
                    se,
                    te,
                    args.input_src,
                    args.input_tgt,
                    tmp_filtered_src,
                    tmp_filtered_tgt,
                    tmp_removed_src,
                    tmp_removed_tgt,
                    args.source_lang,
                    args.target_lang,
                    rank,
                )
                for rank, (se, te) in enumerate(zip(src_edges, tgt_edges))
            ]
        )
    cat_results(
        [args.output_src, args.output_tgt, args.removed_src, args.removed_tgt],
        [tmp_filtered_src, tmp_filtered_tgt, tmp_removed_src, tmp_removed_tgt]
    )
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
