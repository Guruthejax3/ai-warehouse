-- ReplayTwin — PostgreSQL Schema
-- Warehouse Video Intelligence Pipeline

-- Events table: one row per detected unsafe behavior
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    timestamp_sec FLOAT NOT NULL,
    frame_idx INT NOT NULL,
    source_video TEXT NOT NULL,
    behavior_class TEXT NOT NULL,
    risk_score FLOAT NOT NULL,
    risk_level TEXT NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    event_type TEXT,  -- from MotionDetector: sustained | sudden_spike | none
    motion_score FLOAT,
    dtw_distance FLOAT,
    confidence FLOAT,
    physics_valid BOOLEAN NOT NULL DEFAULT false,
    physics_severity FLOAT,
    justification TEXT NOT NULL DEFAULT '',
    evidence_clip_start INT,
    evidence_clip_end INT,
    evidence_clip_path TEXT,
    activity_type TEXT,             -- loading | unloading | idle | transit
    sequence_state TEXT,            -- current FSM step for sequence detection
    shift_id TEXT,                  -- e.g. "shift_20260910_morning"
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_events_behavior ON events(behavior_class);
CREATE INDEX IF NOT EXISTS idx_events_risk ON events(risk_score);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_events_level ON events(risk_level);
CREATE INDEX IF NOT EXISTS idx_events_zone ON events(zone_id);
CREATE INDEX IF NOT EXISTS idx_events_activity ON events(activity_type);

-- Trajectories table: one row per tracked object per event
CREATE TABLE IF NOT EXISTS trajectories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID REFERENCES events(id) ON DELETE CASCADE,
    track_id INT NOT NULL,
    class_label TEXT DEFAULT 'object',
    points JSONB NOT NULL,  -- [{frame_idx, timestamp_sec, x, y, z_est, vx, vy}, ...]
    total_displacement FLOAT,
    mean_speed FLOAT,
    max_speed FLOAT
);

CREATE INDEX IF NOT EXISTS idx_trajectories_event ON trajectories(event_id);

-- Physics states: one per tracked object per event
CREATE TABLE IF NOT EXISTS physics_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID REFERENCES events(id) ON DELETE CASCADE,
    track_id INT NOT NULL,
    peak_velocity FLOAT,
    max_acceleration FLOAT,
    max_jerk FLOAT,
    drop_height_est FLOAT,
    collision_detected BOOLEAN DEFAULT false,
    collision_severity FLOAT,
    severity_score FLOAT,
    justification TEXT
);

-- Exemplars: reference trajectories for behavior matching
CREATE TABLE IF NOT EXISTS exemplars (
    id TEXT PRIMARY KEY,
    behavior_class TEXT NOT NULL,
    label TEXT NOT NULL,
    trajectory_data JSONB NOT NULL,
    source_clip TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_exemplars_behavior ON exemplars(behavior_class);

-- Feedback: human annotator corrections
CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID REFERENCES events(id) ON DELETE CASCADE,
    annotator TEXT,
    correct_behavior TEXT,
    correct_risk_level TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_feedback_event ON feedback(event_id);
