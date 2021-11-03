import sys

def read_text(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()

        return lines

def main(ref_fn, hyp_fn):
    refs = read_text(ref_fn)
    hyps = read_text(hyp_fn)

    correcnt_cnt = 0
    for ref, hyp in zip(refs, hyps):
        if ref == hyp:
            correcnt_cnt += 1
    
    print('%d / %d = %.4f' % (correcnt_cnt, len(refs), float(correcnt_cnt) / len(refs)))

if __name__ == '__main__':
    ref_fn = sys.argv[1]
    hyp_fn = sys.argv[2]

    main(ref_fn, hyp_fn)
