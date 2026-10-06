set -e
D=data/raw
mkdir -p $D
for i in 000 001 002 003 004; do
  gh api "repos/buffedlizard55-lab/GEMSDOE/contents/data/bridge/gems-geodawn-numerical-features.tif.part-$i?ref=c0c06ac82178f26b94fce3397036ef8f12a2f3a0" -H "Accept: application/vnd.github.raw" > "$D/_part-$i"
  echo "part-$i $(stat -c%s $D/_part-$i)"
done
cat $D/_part-000 $D/_part-001 $D/_part-002 $D/_part-003 $D/_part-004 > $D/training_features.tif
rm -f $D/_part-*
sha256sum $D/training_features.tif
stat -c%s $D/training_features.tif
