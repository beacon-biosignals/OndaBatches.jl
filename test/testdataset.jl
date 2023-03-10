using AWSS3
using Legolas: @schema, @version
using Onda

# pointers to test dataset tables + necessary schemas
const signals_path = S3Path("s3://beacon-public-oss/ondabatches-ci/test-data/clean-sleep/test.onda.signal.arrow?versionId=BCcLcHSQPE.RXFstxDcYciWotittBEGE")
const uncompressed_signals_path = S3Path("s3://beacon-public-oss/ondabatches-ci/test-data/clean-sleep/uncompressed.test.onda.signal.arrow?versionId=0eW58zErHNUUjkExSLqpCv24UDwnBDTV")
const stages_path = S3Path("s3://beacon-public-oss/ondabatches-ci/test-data/clean-sleep/test.clean-sleep.sleepstage.arrow?versionId=FiRWymDsbNbeUDFeyWgLmtY8rBPzKTeN")

@schema "sleep-stage" SleepStage
@version SleepStageV1 > AnnotationV1 begin
    stage::String
end
