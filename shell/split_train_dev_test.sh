ls *.json |sort -R |tail -21 |while read file; do
   mv $file ./dev
done

ls *.json |sort -R |tail -21 |while read file; do
   mv $file ./test
done

ls *.json |while read file; do
   mv $file ./train
done
