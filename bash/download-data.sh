temp=./temp
zip_name=multi-label-data.zip
mkdir $temp

wget --load-cookies $temp/cookies.txt "https://docs.google.com/uc?export=download&confirm=\
    $(wget --quiet --save-cookies $temp/cookies.txt --keep-session-cookies --no-check-certificate \
    'https://docs.google.com/uc?export=download&id=FILEID' -O- | \
    sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AKe8gXUh6xlKfgcXO_vINBDvkEAw02xC" \
    -O $zip_name && rm -rf $temp

unzip -q $zip_name

rm -f $zip_name
