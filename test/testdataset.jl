using AWSS3
using Legolas: @row
using Onda

# pointers to test dataset tables + necessary schemas

const signals_path = S3Path("s3://beacon-public-oss/ondabatches-ci/test-data/clean-sleep/test.onda.signal.arrow?versionId=DStLd1W5dPTN4.SBhcbVyirvDun1j0L6")
const uncompressed_signals_path = S3Path("s3://beacon-public-oss/ondabatches-ci/test-data/clean-sleep/uncompressed.test.onda.signal.arrow?versionId=DKQt6r__GOWh9iWvLfxSOg8OMntc0faG")
const stages_path = S3Path("s3://beacon-public-oss/ondabatches-ci/test-data/clean-sleep/test.clean-sleep.sleepstage.arrow?versionId=JoEyaXMcQ0aKKQu12n3ZYOZ9jcHhiF17")

const SleepStageAnnotation = @row("sleep-stage@1" > "onda.annotation@1",
                                  stage::String = validate_sleep_stage(stage))
