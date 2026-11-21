"""Serve the local 2048 game with the trained N-tuple agent."""

from __future__ import annotations

import argparse
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from game2048 import Game2048
from ntuple_agent import MAX_TUPLE_EXPONENT, NtupleAgent


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>2048 RL Agent</title>
  <style>
    :root { color-scheme: light; --ink:#20312b; --muted:#65766f; --paper:#faf8f2;
      --board:#b9ada0; --empty:#cec5ba; --green:#517a61; }
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; background:var(--paper); color:var(--ink);
      font-family:system-ui,-apple-system,Segoe UI,sans-serif; display:flex;
      justify-content:center; padding:28px 16px; }
    main { width:min(520px,100%); }
    header { display:flex; justify-content:space-between; align-items:end; gap:12px; }
    h1 { margin:0; font-size:clamp(2rem,8vw,3.2rem); letter-spacing:-.06em; }
    .tag { color:var(--muted); font-size:.85rem; text-align:right; }
    .intro { color:var(--muted); line-height:1.45; margin:8px 0 18px; }
    .stats { display:flex; gap:8px; margin-bottom:12px; }
    .stat { flex:1; border-radius:10px; background:#e7dfd4; padding:8px 12px;
      text-align:center; }
    .stat small { display:block; text-transform:uppercase; letter-spacing:.08em;
      color:var(--muted); font-size:.68rem; }
    .stat strong { font-size:1.2rem; }
    #board { display:grid; grid-template-columns:repeat(4,1fr); gap:9px; padding:10px;
      background:var(--board); border-radius:12px; aspect-ratio:1; touch-action:manipulation; }
    .cell { display:flex; align-items:center; justify-content:center; min-width:0;
      border-radius:8px; background:var(--empty); color:#51483f; font-size:clamp(1.15rem,7vw,2rem);
      font-weight:750; }
    .tile-2 { background:#eee4da; } .tile-4 { background:#ede0c8; }
    .tile-8 { background:#f2b179;color:#fff; } .tile-16 { background:#f59563;color:#fff; }
    .tile-32 { background:#f67c5f;color:#fff; } .tile-64 { background:#f65e3b;color:#fff; }
    .tile-high { background:#edc850;color:#fff; font-size:clamp(1rem,6vw,1.75rem); }
    .controls { display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 8px; }
    button { border:0; border-radius:9px; padding:11px 14px; background:var(--green);
      color:white; font:inherit; font-weight:650; cursor:pointer; }
    button.secondary { background:#ded7ce; color:var(--ink); }
    button:disabled { opacity:.5; cursor:default; }
    #status { min-height:1.5em; margin:8px 0; color:var(--green); font-weight:650; }
    .help { color:var(--muted); font-size:.88rem; line-height:1.5; }
    code { background:#eee8df; border-radius:4px; padding:1px 4px; }
  </style>
</head>
<body>
<main>
  <header><h1>2048</h1><div class="tag">N-tuple RL<br>score-focused agent</div></header>
  <p class="intro">Play with arrow keys or WASD, take an AI move, or watch the trained model play.</p>
  <section class="stats">
    <div class="stat"><small>Score</small><strong id="score">0</strong></div>
    <div class="stat"><small>Best tile</small><strong id="best">0</strong></div>
    <div class="stat"><small>Moves</small><strong id="moves">0</strong></div>
  </section>
  <div id="board" role="grid" aria-label="2048 board"></div>
  <div class="controls">
    <button id="ai-step">AI move</button>
    <button id="autoplay">Watch AI play</button>
    <button id="pause" class="secondary" disabled>Pause</button>
    <button id="reset" class="secondary">New game</button>
  </div>
  <div id="status" role="status"></div>
  <p class="help">The agent was trained in a local simulator. This page uses the trained model to choose moves; arrow keys let you play the same game yourself.</p>
</main>
<script>
  const boardElement = document.querySelector('#board');
  const statusElement = document.querySelector('#status');
  let autoplay = false;
  let state = null;

  async function request(path, body) {
    const response = await fetch(path, body === undefined ? {} : {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Request failed');
    return payload;
  }

  function draw(nextState) {
    state = nextState;
    boardElement.replaceChildren();
    for (const row of state.board) {
      for (const value of row) {
        const cell = document.createElement('div');
        cell.className = 'cell';
        if (value) {
          cell.textContent = String(value);
          cell.classList.add(value <= 64 ? `tile-${value}` : 'tile-high');
        }
        cell.setAttribute('role', 'gridcell');
        boardElement.append(cell);
      }
    }
    document.querySelector('#score').textContent = state.score.toLocaleString();
    document.querySelector('#best').textContent = state.max_tile.toLocaleString();
    document.querySelector('#moves').textContent = state.moves.toLocaleString();
    if (state.won && !state.game_over) statusElement.textContent = '2048 reached — keep playing for a higher tile.';
    else if (state.won) statusElement.textContent = state.last_actor === 'agent'
      ? `The agent reached 2048; final tile: ${state.max_tile.toLocaleString()}.`
      : `You reached 2048; final tile: ${state.max_tile.toLocaleString()}.`;
    else if (state.done) statusElement.textContent = 'No moves left. Start a new game.';
    else statusElement.textContent = state.last_action ? `Last move: ${state.last_action}` : '';
    document.querySelector('#ai-step').disabled = state.done || autoplay;
    document.querySelector('#autoplay').disabled = state.done || autoplay;
    document.querySelector('#pause').disabled = !autoplay;
  }

  async function doMove(direction) {
    if (!state || state.done || autoplay) return;
    try { draw(await request('/api/move', {action: direction})); }
    catch (error) { statusElement.textContent = error.message; }
  }

  async function aiMove() {
    if (!state || state.done) return;
    draw(await request('/api/ai-move', {}));
  }

  async function watchLoop() {
    if (!autoplay || !state || state.done) {
      autoplay = false;
      if (state) draw(state);
      return;
    }
    try {
      await aiMove();
      if (autoplay && !state.done) window.setTimeout(watchLoop, 140);
      else { autoplay = false; draw(state); }
    } catch (error) {
      autoplay = false;
      statusElement.textContent = error.message;
      draw(state);
    }
  }

  document.addEventListener('keydown', event => {
    const keyMap = {ArrowUp:'up', w:'up', W:'up', ArrowRight:'right', d:'right', D:'right',
      ArrowDown:'down', s:'down', S:'down', ArrowLeft:'left', a:'left', A:'left'};
    const direction = keyMap[event.key];
    if (direction) { event.preventDefault(); doMove(direction); }
  });
  document.querySelector('#ai-step').addEventListener('click', () => aiMove().catch(e => statusElement.textContent=e.message));
  document.querySelector('#autoplay').addEventListener('click', () => { autoplay = true; watchLoop(); });
  document.querySelector('#pause').addEventListener('click', () => { autoplay = false; draw(state); });
  document.querySelector('#reset').addEventListener('click', async () => {
    autoplay = false;
    try { draw(await request('/api/reset', {})); }
    catch (error) { statusElement.textContent = error.message; }
  });
  request('/api/state').then(draw).catch(error => statusElement.textContent = error.message);
</script>
</body>
</html>"""


class GameService:
    def __init__(self, checkpoint: Path, search_depth: int) -> None:
        self.agent = NtupleAgent(checkpoint, search_depth=search_depth)
        self.env = Game2048(
            seed=random.randrange(2**32),
            continue_after_win=True,
        )
        self.last_action: str | None = None
        self.last_actor: str | None = None
        self.lock = threading.RLock()

    def state(self) -> dict[str, Any]:
        legal_actions = () if self.env.done else self.env.valid_actions()
        return {
            "board": self.env.board,
            "score": self.env.score,
            "max_tile": max((value for row in self.env.board for value in row), default=0),
            "moves": self.env.moves,
            "won": self.env.won,
            "game_over": self.env.game_over,
            "done": self.env.done,
            "legal_actions": [Game2048.ACTIONS[index] for index in legal_actions],
            "last_action": self.last_action,
            "last_actor": self.last_actor,
        }

    def reset(self) -> dict[str, Any]:
        self.env = Game2048(
            seed=random.randrange(2**32),
            continue_after_win=True,
        )
        self.last_action = None
        self.last_actor = None
        return self.state()

    def move(self, action: int | str, actor: str = "human") -> dict[str, Any]:
        self.env.step(action)
        self.last_action = (
            Game2048.ACTIONS[action] if isinstance(action, int) else action.lower()
        )
        self.last_actor = actor
        return self.state()

    def ai_move(self) -> dict[str, Any]:
        if self.env.done:
            return self.state()
        legal_actions = self.env.valid_actions()
        if not legal_actions:
            return self.state()
        observation = self.env.observation(max_exponent=MAX_TUPLE_EXPONENT)
        action = self.agent.select_action(observation, legal_actions, epsilon=0.0)
        return self.move(action, actor="agent")


def make_handler(service: GameService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "2048-RL-Agent/1.0"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/state":
                with service.lock:
                    self._send_json(200, service.state())
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                payload = self._read_payload()
                with service.lock:
                    if path == "/api/reset":
                        response = service.reset()
                    elif path == "/api/move":
                        action = payload.get("action")
                        if not isinstance(action, (int, str)) or isinstance(action, bool):
                            raise ValueError("action must be a direction or action id")
                        response = service.move(action)
                    elif path == "/api/ai-move":
                        response = service.ai_move()
                    else:
                        self._send_json(404, {"error": "not found"})
                        return
                self._send_json(200, response)
            except (json.JSONDecodeError, ValueError, RuntimeError, TypeError) as error:
                self._send_json(400, {"error": str(error)})

        def _read_payload(self) -> dict[str, Any]:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("invalid Content-Length") from error
            if content_length < 0 or content_length > 2_048:
                raise ValueError("request body is too large")
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format_string % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/ntuple_best.npz"))
    parser.add_argument("--search-depth", type=int, choices=(1, 2), default=2)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    service = GameService(args.checkpoint, args.search_depth)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service))
    print(f"2048 RL Agent ready at http://127.0.0.1:{args.port}")
    print("Press Ctrl+C in this terminal to stop the local game server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping 2048 RL Agent.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
