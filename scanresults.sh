#!/bin/bash

function scan()
{

    ls trainlog.o* | while read RUN; do
	echo "---$RUN---"
	cat $RUN |tr '\r' '\n'|grep "CFG"
	cat $RUN |tr '\r' '\n'|grep "val_precision" |tail -2
	PREC=$(cat $RUN |tr '\r' '\n'|grep "val_precision" |tail -1 |rev |cut -d":" -f1 | rev)
	echo -e "RESULT:$PREC\t$RUN"
 done
}

echo "Worst in club"
scan |grep -E "^RESULT:" |cut -d":" -f2-  |sort -g |head -2
echo "Best in club"
scan |grep -E "^RESULT:" |cut -d":" -f2-  |sort -gr |head -5
echo "HISTOGRAM"
scan |grep -E "^RESULT:" |cut -d":" -f2-  | cut -b -5 |sort |uniq -c 
