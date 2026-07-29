# VisionUse
### Open Source AI Video Intelligence & Editing Framework

Version: v0.2.0
Status: Project Constitution

---

# Vision

VisionUse is an open-source framework that enables AI agents to understand, analyse, edit, and enhance videos autonomously.

VisionUse is **not a video editor**.

VisionUse is **not another FFmpeg wrapper**.

VisionUse is an intelligent orchestration framework that provides reusable skills for AI agents, similar to how browser-use provides browser capabilities.

VisionUse should become the standard ecosystem for AI-powered video workflows.

---

# Mission

Allow any AI model to perform professional-quality video operations through reusable skills.

Instead of:

> Stabilization = 37

Users should simply say:

> Make this footage look like it was shot on a DJI Ronin.

or

> Remove the camera shake but preserve the handheld cinematic feel.

or

> Convert this talking-head video into an engaging YouTube Short.

VisionUse determines the best approach automatically.

---

# Philosophy

VisionUse follows the same philosophy as browser-use and video-use.

AI Agent

↓

VisionUse Runtime

↓

Skills

↓

Existing Computer Vision / AI Libraries

↓

Final Video

VisionUse orchestrates.

It does not reinvent.

Whenever a mature open-source solution exists, VisionUse should integrate it instead of replacing it.

---

# Goals

- Open Source
- AI Native
- Local First
- Plugin First
- Agent Friendly
- Framework over Application
- Modular
- Production Ready
- Extensible
- Community Driven

---

# Non Goals

VisionUse is NOT:

- a replacement for DaVinci Resolve
- a timeline editor
- another GUI editor
- a monolithic application
- cloud dependent
- proprietary

---

# Technology Stack

Primary Language

Python 3.12+

Reason:

- Same ecosystem as browser-use
- Same ecosystem as video-use
- Largest AI ecosystem
- Largest Computer Vision ecosystem
- Easy community contributions

Core Libraries

OpenCV

FFmpeg

NumPy

PyTorch

ONNX Runtime

MoviePy (only where appropriate)

PyAV

Transformers

Diffusers

Pydantic

Typer

Rich

uv

Testing

pytest

ruff

mypy

GitHub Actions

---

# Repository Structure

vision-use/

    vision_use/

        runtime/

        pipeline/

        skills/

        plugins/

        context/

        validators/

        models/

        ai/

        video/

        cli/

        utils/

    examples/

    docs/

    tests/

    scripts/

    pyproject.toml

---

# Runtime Architecture

User

↓

LLM

↓

Vision Agent

↓

VisionUse Runtime

↓

Skill Manager

↓

Pipeline

↓

Video Output

The runtime should be model agnostic.

Compatible with:

- Claude Code
- Codex
- OpenAI
- Gemini
- Local LLMs
- MCP Clients

---

# Skill Architecture

Every capability is a skill.

Examples

Stabilize

Enhance

Upscale

Subtitle

Color Grade

Trim

Crop

Object Removal

Scene Detection

Frame Interpolation

Export

Denoise

Blur

Motion Tracking

Future skills should require no changes to the runtime.

---

# Skill Interface

Every skill must expose the same lifecycle.

analyse()

↓

plan()

↓

execute()

↓

validate()

↓

score()

↓

return

Every skill should be composable with every other skill.

---

# Pipeline

Video

↓

Decode

↓

Extract Metadata

↓

Analyse

↓

Generate Context

↓

Execute Skill

↓

Validate

↓

Retry (optional)

↓

Export

---

# Context System

Every skill receives the same context.

Example

Video Metadata

FPS

Resolution

Codec

Frame Count

Detected Camera Motion

Detected Scene Type

Detected Subjects

Existing Timeline

User Prompt

Previous Skill Results

Future skills must build upon this context instead of reanalysing the video.

---

# Validation Philosophy

Every skill validates its own output.

If validation fails:

Retry

↓

Alternative algorithm

↓

Alternative parameters

↓

Return highest scoring result

The framework should optimise automatically.

---

# Skill Categories

## Analysis

scene-detection

camera-motion

object-detection

ocr

speech-analysis

face-detection

quality-analysis

---

## Enhancement

stabilize

denoise

sharpen

upscale

hdr

deblur

---

## Editing

trim

crop

silence-removal

jump-cut

subtitle

captions

overlay

transition

---

## AI

object-removal

background-replacement

inpainting

style-transfer

voice-enhancement

video-generation

---

## Export

youtube

instagram

linkedin

tiktok

custom

---

# Initial Skill

Only one skill should be implemented initially.

stabilize

Everything else comes later.

The objective is to create the best open-source stabilization skill available.

---

# Stabilize Skill

Modes

Auto

Walking

Drone

Vehicle

Sports

Tripod

Handheld

Cinematic

Minimal

Aggressive

Future

AI Generated Camera Paths

---

# Motion Analysis

Phase 1

Lucas Kanade

ORB

FAST

Affine Transform

Kalman Filter

Moving Average

Phase 2

RAFT

GMFlow

UniMatch

SAM2

---

# Validation Metrics

Camera Shake

Crop Percentage

Blur Increase

Temporal Consistency

Rolling Shutter Estimate

Motion Smoothness

Frame Distortion

Overall Quality Score

---

# AI Layer

The AI model should never manipulate pixels directly.

The AI model should:

Analyse

↓

Choose Skill

↓

Choose Strategy

↓

Choose Parameters

↓

Review Output

↓

Retry if necessary

All heavy processing should happen inside the framework.

---

# CLI

vision stabilize input.mp4

vision analyze input.mp4

vision score before.mp4 after.mp4

vision plugins

vision doctor

vision benchmark

vision version

---

# Python API

Example

agent = VisionAgent()

result = await agent.run(

    video="video.mp4",

    prompt="""
    Remove the camera shake while preserving the cinematic movement.
    """

)

---

# Plugin System

Plugins should register themselves automatically.

Every plugin must expose:

metadata

capabilities

dependencies

supported formats

configuration schema

No runtime modification should be necessary.

---

# MCP Support

VisionUse should expose an MCP server.

Tools

analyze_video

stabilize_video

score_video

list_skills

execute_skill

Future AI agents should use VisionUse through MCP.

---

# Future Integrations

browser-use

video-use

OpenAI Agents

Claude Code

Codex

LangGraph

CrewAI

AutoGen

MCP

---

# Coding Standards

Python 3.12+

Fully typed

ruff

pytest

mypy

Pydantic

Dependency Injection

No global state

Async where beneficial

Production logging

Comprehensive documentation

100% reproducible builds

---

# Development Roadmap

v0.1

Framework

CLI

Skill API

Pipeline

Configuration

v0.2

Stabilize Skill

Validation

Metrics

v0.3

Plugin SDK

MCP Server

v0.4

AI Decision Engine

v0.5

Video Context Engine

v0.6

Community Skills

v0.7

Cloud Workers (optional)

v1.0

Stable Release

---

# Deliverables

The coding agent should build:

✓ Python project using uv

✓ Modern package structure

✓ Runtime

✓ Skill SDK

✓ Plugin system

✓ Pipeline engine

✓ Stabilization skill

✓ Validation framework

✓ Metrics engine

✓ CLI

✓ Configuration

✓ Logging

✓ Tests

✓ Benchmarks

✓ Documentation

✓ GitHub Actions

✓ Docker

✓ MCP Server

The codebase should be designed as a long-term open-source ecosystem that allows the community to develop and publish independent VisionUse skills without modifying the core framework.