# visual system

The interface has one job: make a dubious piece of machine perception look inspectable rather than magical.

## rules

- lowercase copy;
- black, off-white, and measured greys only;
- no gradients, glow, scan lines, pulsing status dots, or decorative charts;
- animation only when it communicates recording or state change;
- every number on screen must come from the camera, browser, or model;
- uncertainty stays visible as ranked hypotheses;
- prose is short enough to read while looking at your own mouth.

## tokens

| role | value |
|---|---|
| background | `#000000` |
| panel | `#080808` |
| text | `#f2f2ef` |
| muted | `#8a8a86` |
| divider | `#262626` |
| body | Inter / system sans |
| telemetry | SF Mono / system mono |

## hierarchy

1. camera and mouth lock;
2. predicted sentence;
3. alternative readings;
4. capture quality and hardware facts;
5. engine state.

If an element does not help capture, decode, inspect, or recover from failure, it does not ship.
