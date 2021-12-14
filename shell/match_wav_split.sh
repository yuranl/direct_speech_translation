ls ./dev/*.json | while read file; do
   mv "./audio_train/${file:6:(-4)}wav" ./dev/audio
done

ls ./test/*.json | while read file; do
   mv "./audio_train/${file:6:(-4)}wav" ./test/audio
done

ls ./train/*.json | while read file; do
   mv "./audio_train/${file:8:(-4)}wav" ./train/audio
done
