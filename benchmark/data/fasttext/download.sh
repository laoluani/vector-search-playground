set -e


# 1 million word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens).
URL=https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
FASTTEXT_FILENAME=benchmark/data/fasttext/fasttext.vec

if [ ! -f $FASTTEXT_FILENAME ]
then
    curl $URL --output $FASTTEXT_FILENAME.zip
    unzip $FASTTEXT_FILENAME.zip -d benchmark/data/fasttext/
    mv benchmark/data/fasttext/wiki-news-300d-1M.vec $FASTTEXT_FILENAME
fi

# Run the parsing script


python benchmark/data/fasttext/parse.py

