#!/bin/sh
# Deploy the committed HEAD to InstaCloud as a single container (UI on 8502).
#
# `insta deploy` builds the Dockerfile's last stage and has no --target flag, so
# this stages `git archive HEAD` in a temp dir and appends an `instacloud` stage
# built on the upstream `single` target. The repo's own Dockerfile is untouched.
#
# One-time setup (see the PR description): a compute service `app` with a /data
# volume, and secrets OPEN_NOTEBOOK_ENCRYPTION_KEY, OPEN_NOTEBOOK_PASSWORD and
# SURREAL_* set with `insta secrets set`, plus API_URL=https://<service host>:
# InstaCloud exposes only port 8502, so the auto-detected <host>:5055 API URL
# is unreachable and the UI renders blank; the frontend proxies /api itself.
set -eu

ROOT=$(git rev-parse --show-toplevel)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

git -C "$ROOT" archive HEAD | tar -x -C "$STAGE"
cat "$ROOT/scripts/instacloud/Dockerfile.stage" >> "$STAGE/Dockerfile"

cd "$ROOT"
insta deploy "$STAGE" --group "${INSTA_GROUP:-app}" --port 8502 "$@"
