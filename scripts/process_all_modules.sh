#!/bin/bash
# Script to process all modules using tito commands
# Notebooks (.ipynb) are the source of truth - they already exist
# Usage: ./scripts/process_all_modules.sh [start_module] [end_module]

set -e  # Exit on error

# Default range: modules 1-20
START_MODULE=${1:-1}
END_MODULE=${2:-20}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  TinyTorch Module Processing Pipeline                     ║${NC}"
echo -e "${BLUE}║  Notebooks (.ipynb) are source of truth                   ║${NC}"
echo -e "${BLUE}║  Processing modules ${START_MODULE} through ${END_MODULE}                              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Track results
SUCCESSFUL_MODULES=()
FAILED_MODULES=()

# Process each module
for i in $(seq $START_MODULE $END_MODULE); do
    MODULE_NUM=$(printf "%02d" $i)

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}📦 Processing Module ${MODULE_NUM}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Complete module (converts .ipynb to .py and runs tests)
    echo -e "${YELLOW}🔄 Running tito module complete ${MODULE_NUM}...${NC}"

    if tito module complete $MODULE_NUM --skip-export > /tmp/module_${MODULE_NUM}_complete.log 2>&1; then
        echo -e "${GREEN}✅ Module completed successfully!${NC}"
        SUCCESSFUL_MODULES+=($MODULE_NUM)
    else
        echo -e "${RED}❌ Module completion failed!${NC}"
        echo -e "${YELLOW}   View logs: /tmp/module_${MODULE_NUM}_complete.log${NC}"
        # Show last 10 lines of error
        echo -e "${RED}   Last errors:${NC}"
        tail -10 /tmp/module_${MODULE_NUM}_complete.log | sed 's/^/   /'
        FAILED_MODULES+=($MODULE_NUM)
    fi

    echo ""
done

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Processing Complete - Summary                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}✅ Successful (${#SUCCESSFUL_MODULES[@]}): ${SUCCESSFUL_MODULES[*]}${NC}"
echo -e "${RED}❌ Failed (${#FAILED_MODULES[@]}): ${FAILED_MODULES[*]}${NC}"
echo ""

# Exit with error if any failed
if [ ${#FAILED_MODULES[@]} -gt 0 ]; then
    echo -e "${RED}Some modules failed. Check logs in /tmp/module_*_complete.log${NC}"
    exit 1
else
    echo -e "${GREEN}🎉 All modules processed successfully!${NC}"
    exit 0
fi
