import org.jetbrains.kotlin.gradle.plugin.KotlinPluginWrapper

plugins {
    kotlin("jvm") version "1.4-M1"
    application
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    maven("https://dl.bintray.com/kotlin/kotlin-eap")
    mavenCentral()
    maven("https://oss.sonatype.org/content/repositories/snapshots/")
}

dependencies {
    val versions = object {
        val Kotlin by lazy { plugins.getPlugin(KotlinPluginWrapper::class).kotlinPluginVersion }
        val KotlinCoroutines = "1.3.5"
        val Jackson = "2.11.0"
    }

    implementation(kotlin("stdlib-jdk8"))
    //implementation(fileTree( "dir" to "libs", "include" to "*.jar"))
    implementation("ai.djl:api:0.5.0")
    implementation("ai.djl:model-zoo:0.5.0")
    implementation("ai.djl:basicdataset:0.5.0")
    implementation("ai.djl.mxnet:mxnet-engine:0.5.0")
    //implementation("ai.djl.mxnet:mxnet-native-auto:1.6.0")
    implementation("ai.djl.mxnet:mxnet-native-auto:1.7.0-a-SNAPSHOT")
    //implementation("ai.djl.mxnet:mxnet-native-cu102mkl:1.7.0-a-SNAPSHOT")

    implementation("ch.qos.logback", "logback-classic", "1.2.3")

    implementation("com.fasterxml.jackson.core", "jackson-core", versions.Jackson)
    implementation("com.fasterxml.jackson.core", "jackson-databind", versions.Jackson)
    implementation("com.fasterxml.jackson.core", "jackson-annotations", versions.Jackson)
    implementation("com.fasterxml.jackson.module", "jackson-module-kotlin", versions.Jackson)
    implementation("com.fasterxml.jackson.datatype", "jackson-datatype-jsr310", versions.Jackson)
}

tasks {
    compileKotlin {
        kotlinOptions.jvmTarget = "1.8"
    }
    compileTestKotlin {
        kotlinOptions.jvmTarget = "1.8"
    }
}

application {
    mainClassName = "com.waicool20.djl.yolo.YoloRunKt"
}