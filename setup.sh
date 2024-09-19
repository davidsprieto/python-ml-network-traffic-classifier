mkdir -p ~/.streamlit/
# shellcheck disable=SC2028
echo "\
[server]\n\
headless = true\n\
enableCORS=true\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
