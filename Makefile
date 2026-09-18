# Operator shortcuts. `make` with no target prints this help.
IMAGE := vllm/vllm-openai-rocm@sha256:5aa7e626ff73672f5ca7aae46754570488c23d33ca1ac90756a1d2d1a3fe099b

help:  ## show targets
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-12s %s\n",$$1,$$2}'
gate:  ## correctness gate (~21s) - run after EVERY launch
	@bash /scratch/gate.sh
quick: ## tier-1 gate only (~5s)
	@bash /scratch/gate.sh --quick
health: ## endpoint + restart policy + KV
	@echo "http $$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/v1/models)"
	@docker logs k3 2>&1 | grep -E "GPU KV cache size|Maximum concurrency" | tail -2
cache: ## live prefix-cache hit rate (expect ~81%)
	@curl -s 127.0.0.1:8001/metrics | grep -E '^vllm:prefix_cache_(hits|queries)_total' \
	  | awk '{print $$NF}' | paste -sd' ' | awk '{printf "hit_rate=%.2f%%\n",100*$$2/$$1}'
logs:  ## follow server logs
	@docker logs -f --tail 50 k3
sizes: ## CHECK: does an MI355X slug exist on this account? (settles job 1)
	@doctl compute size list --format Slug,Memory,Disk,PriceHourly 2>/dev/null | grep -i mi35 \
	  || echo "no MI355X slug returned -> spot creation is Control-Panel-only"

# ---- repo / deployment targets ----
secret-scan: ## verify no credential is tracked, staged, or in git history
	@bash ./secret-scan.sh
verify: ## security assertions against the live edge (auth + path allowlist)
	@bash ./verify-auth.sh
deploy: ## full idempotent deployment; ./deploy.sh --help for single stages
	@bash ./deploy.sh
plan:  ## show what deploy would do, without doing it
	@bash ./deploy.sh --dry-run
.PHONY: help gate quick health cache logs sizes secret-scan verify deploy plan
