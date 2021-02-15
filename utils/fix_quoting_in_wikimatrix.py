import argparse
import warnings
from pathlib import Path

import regex


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-src",
        "-s",
        help="Path to the input source file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--input-tgt",
        "-t",
        help="Path to the input target file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-src",
        "-S",
        help="Path to the output source file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-tgt",
        "-T",
        help="Path to the output target file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--src-lang",
        "-l",
        help="Language of the source file",
        required=True,
    )
    parser.add_argument(
        "--tgt-lang",
        "-L",
        help="Language of the target file",
        required=True,
    )
    parser.add_argument(
        "--src-before",
        help="Path to a file where source sentences as they were before transformation are saved. Only transformed "
             "sentences are saved.",
        type=Path,
    )
    parser.add_argument(
        "--src-after",
        help="Path to a file where source sentences as they were after transformation are saved. Only transformed "
             "sentences are saved.",
        type=Path,
    )
    parser.add_argument(
        "--tgt-before",
        help="Path to a file where target sentences as they were before transformation are saved. Only transformed "
             "sentences are saved.",
        type=Path,
    )
    parser.add_argument(
        "--tgt-after",
        help="Path to a file where target sentences as they were after transformation are saved. Only transformed "
             "sentences are saved.",
        type=Path,
    )
    args = parser.parse_args()
    args.input_src = args.input_src.expanduser()
    args.input_tgt = args.input_tgt.expanduser()
    args.output_src = args.output_src.expanduser()
    args.output_tgt = args.output_tgt.expanduser()
    if args.src_before is not None:
        args.src_before = args.src_before.expanduser()
    if args.src_after is not None:
        args.src_after = args.src_after.expanduser()
    if args.tgt_before is not None:
        args.tgt_before = args.tgt_before.expanduser()
    if args.tgt_after is not None:
        args.tgt_after = args.tgt_after.expanduser()
    return args


def get_patterns():
    nqt = r'[^"]*[a-zA-Z][^"]*'
    part = fr'(?P<part>(""(?P<additional_quoted>{nqt})"")?(?P<outside_quoted>{nqt}))+'
    with_punctuation = regex.compile(fr'^"{part}(?P<punctuation>( that )|(: )|(, ))""{part}"$')
    no_punctuation = regex.compile(fr'^"(?P<part>{nqt})""(?P<part>{nqt})"')
    said_words_re = regex.compile(
        r"(sa[(id)(ys)y])|(answer[(ed)s]?)|(state[ds]?)|(wrote)|(writes?)(written)|(words were)|(gave an interview)|"
        r"(argue[sd]?)|(announce[sd]?)|(describe[sd]?)|(repl[(ies)(ied)y])|(claim[(ed)s]?)|"
        r"(call[(ed)s]?)|(reads?)|(told)|(tells?)|(quote[sd]?)|(recognize[sd]?)|(noted?)|(reviewe[sd]?)|"
        r"(lament[s(ed)])|(report[(ed)s]?)|(assert[(ed)s]?)|(refuse[sd]?)|(respond[(ed)s]?)|"
        r"(explain[(ed)s]?)|(back[(ed)s]? out)|(was advirtised)|(warn[(ed)s]?)|(was alleged)|"
        r"(display[(ed)s]?[^,:;-]+message)|(issue[ds]?[^,:-;]+statement)|(admit[(ted)s]?)|(declare[sd]?)|"
        r"(express[(ed)(es)]? the view)|(conclude[sd]?)|(comment[(ed)s]?)|(thinks?)|(thought)|(translate[sd]? to)|"
        r"(add[(ed)s]?)|(cast[(ed)s]?[^,:;-]+doubt)|(urge[sd]?)|(offer[(ed)s]?)|(talk[(ed)s]?)|(count[(ed)s]?)|"
        r"(remark[(ed)s]?)|(elaborate[sd]?)|(enusre[sd]?)|(praise[sd]?)|(discuss[(es)(ed)]?)|(appreciate[sd]?)|"
        r"(recommend[(ed)s]?)|(s[iau]ng)|(observe[sd]?)|(mention[s(ed)]?)|(compliment[(ed)s]?)|(cr[y(ied)(ies)]?)|"
        r"(whisper[s(ed)]?)|(grunt[(ed)s]?)|(complain[(ed)s]?)|(regret[s(ted)])|(remember[(red)s]?)|(relate[sd]?)|"
        r"(boast[s(ed)]?)|(confirm[(ed)s]?)|(teach(es)?)|(taught)|(mummble[sd]?)|(insist[(ed)s]?)|(reject[(ed)s]?)|"
        r"(suppose[sd]?)|(testif[y(ied)(ies)]?)|(vote[sd]?)|(reveal[(ed)s]?)|(promote[sd]?)|(speaks?)|(spoken?)|"
        r"(beg[iau]n)|(finish[(es)(ed)]?)|(slip[(ped)s]?[^,:;-]+word)|(emphasize[sd]?)|(demand[(ed)s]?)|(swear)|"
        r"(swor[en])|(ma[(de)(ke)][^,:;-]+[(statement)(remark)])|(spell[s(ed)]?)")
    return {
        "with_punctuation": with_punctuation,
        "no_punctuation": no_punctuation,
        "said_words": said_words_re,
        "double": regex.compile('""'),
        "remove_double_quoting": regex.compile(' *"" *'),
    }


def fix_with_citation_with_punctuation(match, patterns):
    cd = match.capturesdict()
    return patterns['double'].sub('"', cd['part'][0]) \
           + cd['punctuation'][0] \
           + '"' \
           + patterns['double'].sub('"', cd['part'][1]) \
           + '"'


def fix_with_citation_no_punctuation(match, patterns, lang):
    parts = match.capturesdict()["part"]
    if lang == 'en':
        if patterns["said_words"].search(parts[0]):
            res = parts[0].strip() + ', "' + parts[1].strip() + '".'
        elif patterns["said_words"].search(parts[1]):
            res = '"' + parts[0].strip() + '" ' + parts[1].strip()
        else:
            res = parts[0].strip() + ' ' + parts[1].strip()
    else:
        res = parts[0].strip() + ' ' + parts[1].strip()
    return res


def count_front_and_rear_quotes(line):
    front = 0
    while line[front] == '"':
        front += 1
    if line[-4:] == '""."':
        rear = 3
    else:
        rear = 0
    while line[-1 - rear] == '""':
        rear += 1
    return front, rear


def fix_split_quotes(line, patterns):
    n_front, n_rear = count_front_and_rear_quotes(line)
    if n_front == 0:
        warnings.warn(f"The line of the unknown type: '{line}'")
        res = line.replace('"', '')
    elif n_front == 1:
        res = patterns['double'].sub('"', line[1:-1])
    elif n_front == 2:
        warnings.warn(f"The line of the unknown type: '{line}'")
        res = line.replace('"', '')
    elif n_front >= 3:
        double_count = line.count('""', 3, -4)
        if double_count == 0:
            res = line[3:-4] + '.'
        elif double_count % 2 > 0:
            res = line.replace('"', '')
        else:
            warnings.warn(f"Unsure about line {line}")
            res = line[3:-4].replace('""', '"') + '.'
    else:
        assert False
    return res


def fix_external_duplication(line, patterns):
    n_front, n_rear = count_front_and_rear_quotes(line)
    n_min = min(n_rear, n_front)
    double_count = line.count('""', n_min, -n_min)
    if double_count % 2:
        res = line.replace('"', '')
    else:
        res = line[n_min:-n_min].replace('""', '"')
    return res


def fix(line, patterns, lang):
    citation_with_punctuation = patterns["with_punctuation"].match(line)
    if citation_with_punctuation:
        res = fix_with_citation_with_punctuation(citation_with_punctuation, patterns)
    else:
        citation_no_punctuation = patterns["no_punctuation"].match(line)
        if citation_no_punctuation:
            res = fix_with_citation_no_punctuation(citation_no_punctuation, patterns, lang)
        elif line[-4:] == '""."':
            res = fix_split_quotes(line, patterns)
        else:
            res = fix_external_duplication(line, patterns)
    return res


def fix_is_needed(line):
    return line[0] == '"' and line[1] == '"'


def main():
    args = get_args()
    out_src_dir = args.output_src.parent
    out_tgt_dir = args.output_tgt.parent
    out_src_tmp = out_src_dir / Path(".tmp_src")
    out_tgt_tmp = out_tgt_dir / Path(".tmp_tgt")
    patterns = get_patterns()
    src_before = None if args.src_before is None else args.src_before.open('w')
    src_after = None if args.src_after is None else args.src_after.open('w')
    tgt_before = None if args.tgt_before is None else args.tgt_before.open('w')
    tgt_after = None if args.tgt_after is None else args.tgt_after.open('w')
    with args.input_src.open() as in_s, args.input_tgt.open() as in_t, out_src_tmp.open('w') as out_s, \
            out_tgt_tmp.open('w') as out_t:
        for s_l, t_l in zip(in_s, in_t):
            s_l, t_l = s_l.strip(), t_l.strip()
            if fix_is_needed(s_l):
                if src_before is not None:
                    src_before.write(s_l)
                    src_before.write('\n')
                s_l = fix(s_l, patterns, args.src_lang)
                if src_after is not None:
                    src_after.write(s_l)
                    src_after.write('\n')
            if fix_is_needed(t_l):
                if tgt_before is not None:
                    tgt_before.write(t_l)
                    tgt_before.write('\n')
                t_l = fix(t_l, patterns, args.tgt_lang)
                if tgt_after is not None:
                    tgt_after.write(t_l)
                    tgt_after.write('\n')
            out_s.write(s_l)
            out_s.write('\n')
            out_t.write(t_l)
            out_t.write('\n')
    if src_before is not None:
        src_before.close()
    if src_after is not None:
        src_after.close()
    if tgt_before is not None:
        tgt_before.close()
    if tgt_after is not None:
        tgt_after.close()
    out_src_tmp.rename(args.output_src)
    out_tgt_tmp.rename(args.output_tgt)


if __name__ == "__main__":
    main()
