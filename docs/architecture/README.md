# Architecture Notes

## Current System Shape

The repository is a single integrated system repo. The project is being built in layers rather than as disconnected subprojects.

Current implementation order:

1. camera input stabilization on Jetson
2. ingredient recognition baseline
3. alcohol recognition baseline
4. inventory or event logic
5. recommendation logic
6. app-facing integration

## External-Facing Camera Assumption

The most important architectural constraint is that the current camera is external-facing.

That means the camera observes:

- ingredients shown while being inserted or removed
- alcohol bottles or cans explicitly shown by the user

It does not currently observe the inside of the fridge as a persistent multi-object scene.

## Current Ingredient Recognition Flow

```text
C920 camera
  -> GStreamer/OpenCV frame capture
  -> frame-level ingredient classifier
  -> confidence threshold + top-k display
  -> candidate ingredient result
```

In the current repo, this corresponds to:

- `scripts/camera/`
- `scripts/data_prep/`
- `src/ingredient_recognition/`

## Future System Direction

The target long-term system flow is:

```text
external camera input
  -> ingredient recognition
  -> alcohol recognition
  -> insert/remove event interpretation
  -> inventory state update
  -> recipe candidate retrieval
  -> recommendation ranking
  -> app or display output
```

## Module Boundaries

- `src/ingredient_recognition/`
  - current completed model-training and inference baseline
- `src/alcohol_recognition/`
  - next recognition module to implement
- `src/inventory/`
  - future state tracking for present and missing items
- `src/recommendation/`
  - future DB-first ranking and fallback generation
- `src/app/`
  - future interface or integration entrypoint

## Design Principle

The project favors:

- small working vertical slices
- Jetson-realistic implementation
- path-safe scripts
- metadata-driven model selection
- incremental extension over large speculative rewrites

