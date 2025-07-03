set -e

# constants
# 1 million word vectors trained on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens).
URL = https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
FASTTEXT_FILENAME=fasttext.vec

if [ ! -f $FASTTEXT_FILENAME ]
then
    curl $URL --output $FASTTEXT_FILENAME.zip
    gzip -d $RAW_FASTTEXT.gz
fi

