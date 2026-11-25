import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { resolve } from "node:path";

const root = resolve(import.meta.dir, "../..");
const python = resolve(root, process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python");

export default function (pi: ExtensionAPI): void {
	pi.on("session_stop", async () => {
		let diagnostics: string;
		try {
			const result = await pi.exec(python, ["-m", "pyright"], { cwd: root, timeout: 20_000 });
			if (result.code === 0 && !result.killed) return;
			diagnostics = [result.stdout.trim(), result.stderr.trim()].filter(Boolean).join("\n");
			if (result.killed) diagnostics = `Проверка прервана или превысила 20 секунд.\n${diagnostics}`;
			else if (!diagnostics) diagnostics = `Код завершения: ${result.code}.`;
		} catch (error) {
			diagnostics = String(error);
		}
		return {
			decision: "block",
			reason: `Pyright не прошёл. Исправь ошибки или восстанови .venv (python -m pip install -e '.[dev]') и повтори проверку.\n${diagnostics}`,
		};
	});
}
