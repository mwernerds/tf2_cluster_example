#!/bin/bash

function loganalyze()
{
    #"echo  {}:; grep precision {} |tr '\r' '\n' | tail -1; cat {} |grep JSONCFG"
    if grep precision "$1" > /dev/null; then
	TESTPREC=$(grep precision "$1" |tr '\r' '\n' | tail -1 |rev | cut -d":" -f1  |rev  )
    else
	TESTPREC=-1
    fi
    grep JSONCFG "$1" | cut -d":" -f2- \
	| jq '. + {"logfile":"'"$1"'"}' \
	| jq '. + {"testprec":'"$TESTPREC"'}' 
	     
}

export -f loganalyze


ls trainlog*  | parallel loganalyze {}
