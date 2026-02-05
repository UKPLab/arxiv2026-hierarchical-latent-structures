use arrow::array::UInt16Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use std::fs::{self, File};
use std::sync::Arc;

pub fn save_u16_as_arrow(path: &str, data: impl Iterator<Item = u16>) -> anyhow::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }

    const CHUNK_SIZE: usize = 1_000_000;

    let schema = Arc::new(Schema::new(vec![Field::new("input_ids", DataType::UInt16, false)]));

    let file = File::create(path)?;
    let mut writer = StreamWriter::try_new(file, &schema)?;

    let mut total_length = 0;
    let mut chunk = Vec::with_capacity(CHUNK_SIZE);

    for item in data {
        chunk.push(item);
        total_length += 1;

        if chunk.len() == CHUNK_SIZE {
            let array = UInt16Array::from_iter(chunk.drain(..));
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)])?;
            writer.write(&batch)?;
        }
    }

    if !chunk.is_empty() {
        let array = UInt16Array::from_iter(chunk.drain(..));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)])?;
        writer.write(&batch)?;
    }

    writer.finish()?;

    let metadata_path = format!("{}.metadata.json", path);
    let metadata_json = serde_json::json!({ "total_length": total_length });
    fs::write(&metadata_path, serde_json::to_string_pretty(&metadata_json)?)?;

    Ok(())
}
