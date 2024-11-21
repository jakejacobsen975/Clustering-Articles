#!/bin/bash
rm -r articles
hadoop jar /home/jake/hadoop-3.4.0/share/hadoop/tools/lib/hadoop-streaming-3.4.0.jar \
    -mapper mapper.py \
    -reducer reducer.py \
    -input Inputfiles/cnbc.json \
    -input Inputfiles/ksl_articles_9-24.json \
    -input Inputfiles/ksl_articles_9-26.json \
    -input Inputfiles/ksl_articles_9-27.json \
    -input Inputfiles/the_verge.json \
    -input Inputfiles/cbs.json \
    -input Inputfiles/abc.json \
    -output articles





