#!/usr/bin/env bash
# Starts a throwaway submilli-server with the `research` Blueprint, for the
# integration tests and the example. Everything it writes lives under
# $SUBMILLI_HOME (default: .submilli-test/ at the workspace root), so it never
# touches a server you already run from ~/.submilli.
#
#   export SUBMILLI_SERVER_TOKEN=$(openssl rand -hex 32)
#   scripts/test-server.sh            # foreground; Ctrl-C stops it
#
# Environment:
#   SUBMILLI_BIN            Directory holding `submilli` and `submilli-server`
#                           (default: .submilli-bin/ at the workspace root if it
#                           exists, otherwise PATH).
#   SUBMILLI_RUNTIME        A submilli-runtime checkout. When set, @submilli/jina is
#                           built from it; otherwise it is installed from GitHub.
#   SUBMILLI_PORT           Port to listen on (default 8128).
#   JINA_API_KEY, ANTHROPIC_API_KEY
#                           Stored as the Blueprint's secrets. Placeholders are
#                           stored when unset; the tests never call Jina or a model.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
root="$(cd "$repo/../.." && pwd)"
[ -z "${SUBMILLI_BIN:-}" ] && [ -x "$root/.submilli-bin/submilli-server" ] && SUBMILLI_BIN="$root/.submilli-bin"
bin="${SUBMILLI_BIN:+$SUBMILLI_BIN/}"
port="${SUBMILLI_PORT:-8128}"
export SUBMILLI_HOME="${SUBMILLI_HOME:-$root/.submilli-test}"
export SUBMILLI_SERVER_URL="http://127.0.0.1:$port"
: "${SUBMILLI_SERVER_TOKEN:?export SUBMILLI_SERVER_TOKEN first}"

mkdir -p "$SUBMILLI_HOME"
[ -f "$SUBMILLI_HOME/store.key" ] || head -c 32 /dev/urandom | base64 > "$SUBMILLI_HOME/store.key"
cat > "$SUBMILLI_HOME/server.yaml" <<EOF
secret_store:
  key_file: $SUBMILLI_HOME/store.key
volumes:
  notes:
    kind: managed-local
    size_limit: unlimited
EOF

if [ -n "${SUBMILLI_RUNTIME:-}" ]; then
  (cd "$SUBMILLI_RUNTIME" && "${bin}submilli" build publish-local -p @submilli/jina)
fi

"${bin}submilli-server" --config "$SUBMILLI_HOME/server.yaml" --port "$port" &
server=$!
trap 'kill $server 2>/dev/null' EXIT INT TERM

for _ in $(seq 1 100); do
  curl -fsS -o /dev/null -H "Authorization: Bearer $SUBMILLI_SERVER_TOKEN" \
    "$SUBMILLI_SERVER_URL/v1/blueprints/research/builtins" 2>/dev/null && break
  # 404 (no blueprint yet) still means the server is up.
  code=$(curl -s -o /dev/null -w '%{http_code}' "$SUBMILLI_SERVER_URL/v1/blueprints/research/builtins" || true)
  [ "$code" != "000" ] && break
  sleep 0.2
done

if [ -z "${SUBMILLI_RUNTIME:-}" ]; then
  "${bin}submilli" server packages install submilli/submilli-runtime @submilli/jina || true
fi
printf '%s' "${JINA_API_KEY:-placeholder}" | "${bin}submilli" server secret put jina_api_key
printf '%s' "${ANTHROPIC_API_KEY:-placeholder}" | "${bin}submilli" server secret put anthropic_api_key
"${bin}submilli" server blueprint apply "$repo/examples/research/blueprint.yaml"

echo "submilli-server is up at $SUBMILLI_SERVER_URL with the research Blueprint"
wait $server
