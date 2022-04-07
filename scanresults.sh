#!/bin/bash

ls trainlog.o* | while read RUN; do
    echo "---$RUN---"
    cat $RUN |tr '\r' '\n'|grep "CFG"
    cat $RUN |tr '\r' '\n'|grep "val_precision" |tail -2
    PREC=$(cat $RUN |tr '\r' '\n'|grep "val_precision" |tail -1 |rev |cut -d":" -f1 | rev)
    echo "RESULT: $RUN == $PREC"
done

