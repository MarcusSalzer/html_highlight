inaliases=$(grep '"[^"]+":' -oE data/class_aliases_str.json |tr -d '":'| sort)
inreadme=$(grep '`.+`' -oE README.md | tr -d '`'| sort)
echo ALIASES: $inaliases
echo README : $inreadme

echo "-------------------------------------------------------"

diff  <(echo "$inaliases") <(echo "$inreadme") --color=always


if [ "$inaliases" = "$inreadme" ]; then
    echo Match!
else
    echo Mismatch!
    exit 1
fi