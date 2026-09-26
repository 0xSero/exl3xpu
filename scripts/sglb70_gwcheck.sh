#!/usr/bin/env bash
# Chat request through the Omarchy gateway (key read from file, never printed).
K=$(cat ~/.local/state/omarchy/local-ai/gateway.key)
M=$(curl -s -m 20 -H "Authorization: Bearer $K" http://127.0.0.1:12434/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
echo "model: $M"
curl -s -m 300 -H "Authorization: Bearer $K" -H "Content-Type: application/json" http://127.0.0.1:12434/v1/chat/completions \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*3? Answer with the number only.\"}],\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('answer:',d['choices'][0]['message']['content'].strip(),'| finish:',d['choices'][0]['finish_reason'])"
