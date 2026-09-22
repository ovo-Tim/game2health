// ==UserScript==
// @name         Game2Health Subway Surfers Bridge
// @namespace    https://github.com/game2health
// @version      1.0.0
// @description  Connects Game2Health to Subway Surfers and automates game lifecycle.
// @match        https://files.gamezhero.com/game/*/*/*/data/index.html*
// @run-at       document-start
// @grant        unsafeWindow
// ==/UserScript==

(() => {
    "use strict";

    // Tampermonkey applies @match, but generic injection (init scripts,
    // broad matches) would otherwise run on the portal page too and fight
    // the game-frame instance for the single bridge client slot.
    if (!/\/data\/index\.html/.test(window.location.href)) {
        return;
    }

    const page = unsafeWindow;
    const BRIDGE_URL = "ws://127.0.0.1:8765";
    const KEY_BY_ACTION = {
        move_left: ["ArrowLeft", "ArrowLeft", 37],
        move_right: ["ArrowRight", "ArrowRight", 39],
        jump: ["ArrowUp", "ArrowUp", 38],
        crouch: ["ArrowDown", "ArrowDown", 40],
    };
    const SPACE = [" ", "Space", 32];

    let socket = null;
    let gameState = "loading";
    let controllerPaused = false;
    let suppressStopUntil = 0;
    let restartGeneration = 0;
    let unityWasReady = false;
    let pendingCommand = null;

    function send(message) {
        if (socket?.readyState === page.WebSocket.OPEN) {
            socket.send(JSON.stringify(message));
        }
    }

    function setState(state) {
        gameState = state;
        send({type: "state", state});
    }

    function connect() {
        // Construct in the page realm so the handshake Origin is the game
        // origin; the desktop bridge rejects extension/foreign origins.
        socket = new page.WebSocket(BRIDGE_URL);
        socket.addEventListener("open", () => {
            send({type: "state", state: gameState});
            if (unityReady()) {
                send({type: "ready", version: 1, game: "subway-zurich"});
            }
        });
        socket.addEventListener("message", (event) => {
            let message;
            try {
                message = JSON.parse(event.data);
            } catch (_error) {
                return;
            }
            if (message.type === "command") {
                handleCommand(message.command);
            } else if (message.type === "actions" && Array.isArray(message.actions)) {
                for (const action of message.actions) {
                    dispatchAction(action);
                }
            }
        });
        socket.addEventListener("close", () => {
            socket = null;
            page.setTimeout(connect, 1000);
        });
        socket.addEventListener("error", () => socket?.close());
    }

    function unityReady() {
        return typeof page.unityGame?.SendMessage === "function";
    }

    function callUnity(objectName, methodName) {
        if (!unityReady()) {
            return false;
        }
        page.unityGame.SendMessage(objectName, methodName);
        return true;
    }

    function dispatchKey([key, code, number]) {
        const options = {
            key,
            code,
            keyCode: number,
            which: number,
            bubbles: true,
            cancelable: true,
        };
        page.document.dispatchEvent(new page.KeyboardEvent("keydown", options));
        page.setTimeout(() => {
            page.document.dispatchEvent(new page.KeyboardEvent("keyup", options));
        }, 35);
    }

    function dispatchAction(action) {
        if (controllerPaused) {
            return;
        }
        const key = KEY_BY_ACTION[action];
        if (key) {
            dispatchKey(key);
        }
    }

    function handleCommand(command) {
        if (!unityReady()) {
            pendingCommand = command;
            controllerPaused = command === "pause";
            return;
        }
        restartGeneration += 1;
        if (command === "start") {
            controllerPaused = false;
            suppressStopUntil = Date.now() + 1500;
            page.focus();
            // Space starts the front screen. StartRun handles an already-open
            // results screen, so a retained start command also recovers after
            // the userscript reconnects mid-session.
            callUnity("GameOverScreen", "StartRun");
            dispatchKey(SPACE);
        } else if (command === "pause") {
            controllerPaused = true;
            suppressStopUntil = Date.now() + 1500;
            callUnity("1PauseButton", "Send");
            setState("paused");
        } else if (command === "resume") {
            controllerPaused = false;
            suppressStopUntil = Date.now() + 2000;
            callUnity(
                "PauseUI/GuideWidget/Footer/3ResumeButton/Button",
                "Send"
            );
        }
    }

    function wrapPokiMethod(name, after) {
        const sdk = page.PokiSDK;
        const original = sdk?.[name];
        if (typeof original !== "function" || original.__g2hWrapped) {
            return;
        }
        function wrapped(...args) {
            const result = original.apply(this, args);
            after();
            return result;
        }
        Object.defineProperty(wrapped, "__g2hWrapped", {value: true});
        sdk[name] = wrapped;
    }

    function installLifecycleHooks() {
        wrapPokiMethod("gameplayStart", () => {
            restartGeneration += 1;
            setState("running");
        });
        wrapPokiMethod("gameplayStop", () => {
            if (controllerPaused || Date.now() < suppressStopUntil) {
                setState("paused");
                return;
            }
            setState("gameover");
            autoRestart();
        });
    }

    function autoRestart() {
        const generation = ++restartGeneration;
        let attempts = 0;
        function attempt() {
            if (generation !== restartGeneration || controllerPaused
                    || gameState === "running") {
                return;
            }
            if (page.document.hidden || !unityReady()) {
                page.setTimeout(attempt, 500);
                return;
            }
            // The first Space advances the New High Score/count-up overlay.
            // Once GameOverScreen becomes active, StartRun bypasses its PLAY
            // button and begins the next run directly.
            dispatchKey(SPACE);
            callUnity("GameOverScreen", "StartRun");
            attempts += 1;
            if (attempts < 20) {
                page.setTimeout(attempt, 750);
            }
        }
        page.setTimeout(attempt, 500);
    }

    page.setInterval(() => {
        installLifecycleHooks();
        if (unityReady() && !unityWasReady) {
            unityWasReady = true;
            setState("ready");
            send({type: "ready", version: 1, game: "subway-zurich"});
            if (pendingCommand !== null) {
                const command = pendingCommand;
                pendingCommand = null;
                handleCommand(command);
            }
        }
    }, 100);

    connect();
})();
