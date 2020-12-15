#!/usr/bin/env bash
set -e

pip3=pip3

echo 'Uninstalling stuff'
${pip3} uninstall -y nemo_toolkit

# Kept for legacy purposes
${pip3} uninstall -y nemo_asr
${pip3} uninstall -y nemo_nlp
${pip3} uninstall -y nemo_tts
${pip3} uninstall -y nemo_simple_gan

${pip3} install -U setuptools

for f in $(ls requirements/*.txt); do ${pip3} install ${pip3_FLAGS}--disable-pip3-version-check --no-cache-dir -r $f; done

echo 'Installing stuff'
${pip3} install -e ".[all]"

echo 'All done!'
