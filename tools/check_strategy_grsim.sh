#!/usr/bin/env bash
# Run one physical regression without changing the interactive PAcmaCS instance.
set -euo pipefail
cd "$(dirname "$0")/.."
scenario_tmp=$(mktemp -d /tmp/ssl-strategy-check.XXXXXX)
scenario_container="ssl-strategy-check-$$"
cleanup() {
    docker stop "$scenario_container" >/dev/null 2>&1 || true
    rm -rf "$scenario_tmp"
}
trap cleanup EXIT
python3 - "${GRSIM_CONFIG:-../PAcmaCS/grsim/.grsim.divC.xml}" "$scenario_tmp/grsim.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET
source, destination = sys.argv[1:]
tree = ET.parse(source)
root = tree.getroot()
communication = root.find("Var[@name='Communication']")
if communication is None:
    communication = ET.SubElement(root, 'Var', name='Communication', type='list')
ports = {'Vision multicast port': 11006, 'Command listen port': 21011,
         'Blue Team status send port': 31011, 'Yellow Team status send port': 31012,
         'Simulation control port': 11300, 'Blue team control port': 11301,
         'Yellow team control port': 11302}
for name, port in ports.items():
    variable = communication.find(f"Var[@name='{name}']")
    if variable is None:
        variable = ET.SubElement(communication, 'Var', name=name, type='int')
    variable.text = str(port)
tree.write(destination, encoding='utf-8', xml_declaration=True)
PY
docker run --rm -d --name "$scenario_container" --network host \
    -v "$scenario_tmp/grsim.xml:/home/default/.grsim.xml" robocupssl/grsim:latest >/dev/null
sleep 2
DIVISION=C IS_SIMULATOR_USED=true COLOR=BLUE POLARITY=1 GK=0 ENEMY_GK=0 \
    venv/bin/python -m tools.run_strategy_scenario --expect-goal --expect-combination "$@"
