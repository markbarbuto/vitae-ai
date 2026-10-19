#!/bin/sh
# Sends each /cases/<name>.hdr through nginx with curl, over IPv4 or IPv6 by the .v4 or .v6 in its
# name: a POST to /v1/chat when <name>.body exists, else GET /_test/key. The raw answer, status
# line and headers included, goes to /out/<name>.resp. key-* cases go 20 at a time, and once more
# one by one if the answer was empty; zz-* cases go one by one, in name order, never retried.
send() {
    base=${1##*/}
    base=${base%.hdr}
    case "$base" in
        *.v6) url="http://[fd00:7a17::2]" ;;
        *) url="http://172.31.250.2" ;;
    esac
    if [ -e "/cases/$base.body" ]; then
        curl -s -i -m 30 -H @"$1" --data-binary @"/cases/$base.body" "$url/v1/chat" > "/out/$base.resp"
    else
        curl -s -i -m 30 -H @"$1" "$url/_test/key" > "/out/$base.resp"
    fi
}
n=0
for headers in /cases/key-*.hdr; do
    send "$headers" &
    n=$((n + 1))
    [ $((n % 20)) -eq 0 ] && wait
done
wait
for headers in /cases/key-*.hdr; do
    base=${headers##*/}
    [ -s "/out/${base%.hdr}.resp" ] || send "$headers"
done
for headers in /cases/zz-*.hdr; do
    if [ -e "$headers" ]; then send "$headers"; fi
done
