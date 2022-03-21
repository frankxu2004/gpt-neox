for LANG in C C++ C# PHP Go Java JavaScript Scala TypeScript Python Ruby Rust
do
    python tools/preprocess_data.py --input testsamples/Code-sampled50/$LANG --tokenizer-type GPT2BPETokenizer --vocab-file data/code-vocab.json --merge-file data/code-merges.txt --output-prefix testsamples/$LANG/code --workers 16
done
