"""Build timeline.js from per-line durations (and optional voice envelope)."""
import json

LEAD = 0.35        # silence before first line
GAP_LINE = 0.08    # pause between lines inside a scene
GAP_SCENE = 0.40   # pause between scenes
TAIL = 1.6         # hold after last line


def layout(script, durs):
    t = LEAD
    scenes = []
    k = 0
    for si, s in enumerate(script['scenes']):
        if si:
            t += GAP_SCENE - GAP_LINE
        lines = []
        for l in s['lines']:
            d = durs[k]; k += 1
            lines.append({'start': round(t, 3), 'end': round(t + d, 3), 'cap': l['cap'], 'tts': l['tts']})
            t += d + GAP_LINE
        scenes.append({'id': s['id'], 'lines': lines})
    total = t - GAP_LINE + TAIL
    for i, s in enumerate(scenes):
        s['vs'] = 0.0 if i == 0 else round(s['lines'][0]['start'] - 0.3, 3)
    for i, s in enumerate(scenes):
        s['ve'] = scenes[i + 1]['vs'] if i + 1 < len(scenes) else round(total + 5, 3)
    return scenes, round(total, 3)


def write_js(path, scenes, total, env, fps):
    with open(path, 'w') as f:
        f.write('window.TL = ' + json.dumps({'fps': fps, 'total': total, 'env': env, 'scenes': scenes}, ensure_ascii=False) + ';\n')
