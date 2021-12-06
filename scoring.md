# Scoring / Evaluating the System
We score our system with BLEU and NIST.
For speech-speech translation, we use an off-the-shelf ASR to recognize the translation and reference, and then score them just like text-text systems.
Follow the instuctions below to score the sytem

## Convert output
For text-text translation, run `json_to_sgm.py`
For speech-speech translation, run `wav_to_sgm.py`

In the "change below parameters" section:
change `input_path` to the folder that contains the input files;
change `mode` to `"ref"`, `"tst"`, `"src"`, respectively to convert reference, MT output, and source files

Produce a `.sgm` file each for reference, MT output, and source

## Run evaluation script
The evalution script is a Perl script `mteval-v13a.pl`
Run:
`perl mteval-v13a.pl -r <refernce>.sgm -s <source>.sgm -t <MT output>.sgm`

## Example input
See `102524.json` for example source and reference
See `102524_result.json` for example MT output, which is produced by our strong baseline

## Example output
> command line:  mteval-v13a.pl -r ref1.sgm -s src1.sgm -t tst1.sgm
>  Evaluation of chs-to-eng translation using:
>    src set "1" (1 docs, 1 segs)
>    ref set "1" (1 refs)
>    tst set "1" (1 systems)
>
> length ratio: 1.004914004914 (4499/4477), penalty (log): 0
> NIST score = 6.2937  BLEU score = 0.1887 for system "tst1"

> # ------------------------------------------------------------------------

> Individual N-gram scoring
>        1-gram   2-gram   3-gram   4-gram   5-gram   6-gram   7-gram   8-gram   9-gram
>        ------   ------   ------   ------   ------   ------   ------   ------   ------
>  NIST:  5.0827   1.0504   0.1405   0.0173   0.0028   0.0008   0.0000   0.0000   0.0000  "tst1"
>
>  BLEU:  0.6913   0.3128   0.1187   0.0494   0.0238   0.0127   0.0071   0.0045   0.0024  "tst1"

> # ------------------------------------------------------------------------
> Cumulative N-gram scoring
>        1-gram   2-gram   3-gram   4-gram   5-gram   6-gram   7-gram   8-gram   9-gram
>        ------   ------   ------   ------   ------   ------   ------   ------   ------
>  NIST:  5.0827   6.1331   6.2735   6.2908   6.2937   6.2945   6.2945   6.2945   6.2945  "tst1"
>
>  BLEU:  0.6913   0.4650   0.2950   0.1887   0.1247   0.0852   0.0598   0.0432   0.0314  "tst1"
