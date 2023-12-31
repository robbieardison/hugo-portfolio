FILES="$(find notebooks -type f -name '*.ipynb')"
for f in $FILES
do
    $f --site-dir base --section post
done
hugo -s base
